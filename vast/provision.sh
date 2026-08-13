#!/usr/bin/env bash
# Provision meetscribe on a vast.ai instance.
#
# PASTE THIS WHOLE FILE into a vast template's "On-start Script" and set nothing
# else. Pin a version by editing MS_REF's default below to a full 40-character
# commit sha; left as `main` it installs whatever main happens to be.
#
# (It also works as a PROVISIONING_SCRIPT url, which is worth using if you want
# several templates sharing one script -- see vast/README.md. Then On-start is
# `exec /opt/instance-tools/bin/entrypoint.sh` instead, and this logs to
# /var/log/portal/provisioning.log.)
#
# It clones the repo, installs the pipeline, downloads the weights, and then
# does one throwaway transcription so the engine's compile cache is populated
# before you ever ask it for anything. That last step is the whole point: the
# FIRST engine load on a machine costs 240-400s while torch.compile and
# FlashInfer fill their caches, against ~70s every load after. Paying it here
# means the box is warm when you arrive rather than when you are waiting.
#
# Safe to run twice. Everything below is either idempotent or guarded.
set -euo pipefail

MS_WORK="${MS_WORK:-/opt/meetscribe}"
MS_REPO="${MS_REPO:-https://github.com/suzuenhasa/meetscribe-cli.git}"
MS_REF="${MS_REF:-main}"
# NOT /workspace: vast documents it as possibly shared between instances with
# concurrent writers, and speakers.db is the one file here that cannot be
# rebuilt from the audio.
export HF_HOME="${HF_HOME:-$MS_WORK/.hf_home}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$MS_WORK/.vllm_cache}"

log()  { printf '[meetscribe %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { printf '[meetscribe] FAILED: %s\n' "$*" >&2; exit 1; }

log "work dir   $MS_WORK"
log "ref        $MS_REF"
log "hf cache   $HF_HOME"

# ------------------------------------------------------------- the entrypoint
# So this file can simply be PASTED into a vast template's On-start Script and
# be the only thing you configure.
#
# SSH launch mode replaces the image's entrypoint with On-start, and that
# entrypoint is what starts supervisord -- which runs the instance portal, the
# tunnel manager, and the engine service registered further down. Pasting a
# script here without starting it therefore silently loses all of them, and the
# instance still boots and accepts ssh, so nothing tells you.
#
# Harmless when vast ran this as PROVISIONING_SCRIPT instead, which is the other
# supported way in: supervisord is already up, and this does nothing.
if ! supervisorctl status >/dev/null 2>&1; then
  if [ -x /opt/instance-tools/bin/entrypoint.sh ]; then
    log "starting the image entrypoint (supervisord, portal, tunnels)"
    /opt/instance-tools/bin/entrypoint.sh >/var/log/entrypoint.log 2>&1 &
    for _ in $(seq 1 30); do
      supervisorctl status >/dev/null 2>&1 && break
      sleep 1
    done
  else
    log "no image entrypoint here; carrying on without supervisord"
  fi
fi

# ---------------------------------------------------------------- the GPU
command -v nvidia-smi >/dev/null || die "no nvidia-smi; this template needs a GPU instance"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | sed 's/^/  /'

# Ray ships started on some vast images and does nothing for us but hold VRAM
# and pids. The stock vllm service refuses to start without VLLM_MODEL, so it is
# already quiet, but stop it too rather than rely on that.
for svc in ray vllm; do
  supervisorctl stop "$svc" >/dev/null 2>&1 && log "stopped the stock $svc service" || true
done

# ---------------------------------------------------------------- the source
if [ -d "$MS_WORK/.git" ]; then
  log "updating the checkout"
else
  log "cloning $MS_REPO"
  mkdir -p "$(dirname "$MS_WORK")"
  # Clone bare-ish then fetch the ref, rather than --branch: --branch takes a
  # branch or tag and NOT a commit sha, and the whole point of pinning MS_REF to
  # a sha is that a moving ref is a thing someone else can change under you. A
  # fallback that quietly cloned the default branch would defeat exactly that.
  git clone --no-checkout --depth 1 "$MS_REPO" "$MS_WORK" || die "git clone"
fi

# One path for both cases, and it accepts a full sha, a tag or a branch.
# ABBREVIATED shas do not work here: git fetch resolves a ref name on the remote
# and an 8-character sha is not one, so it fails with "couldn't find remote ref"
# -- verified against this repo. Say so rather than making someone read that.
if [ "${#MS_REF}" -lt 40 ] && printf '%s' "$MS_REF" | grep -qE '^[0-9a-f]{6,39}$'; then
  die "MS_REF looks like a shortened commit sha ($MS_REF). git fetch needs the
       full 40 characters, or a branch or tag name."
fi
git -C "$MS_WORK" fetch --depth 1 origin "$MS_REF" \
  || die "no such ref in $MS_REPO: $MS_REF"
git -C "$MS_WORK" checkout -f FETCH_HEAD || die "git checkout $MS_REF"
cd "$MS_WORK"
log "at $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"

# So ./transcribe --host <box> from a laptop can find the install without the
# caller having to remember where the template put it.
echo "$MS_WORK" > /etc/meetscribe-work

# ---------------------------------------------------------------- the install
# setup.sh is idempotent and re-runnable; on the vastai/vllm image most of the
# heavy lifting (vLLM, torch, CUDA, ffmpeg) is already done, so this is mostly
# the weights.
log "running setup.sh — weights are ~2 GB on a cold box"
MS_WORK="$MS_WORK" HF_HOME="$HF_HOME" bash setup.sh 2>&1 | sed 's/^/  /' \
  || die "setup.sh — see the output above"

log "verifying"
MS_WORK="$MS_WORK" HF_HOME="$HF_HOME" bash setup.sh --check 2>&1 | sed 's/^/  /' \
  || die "setup.sh --check did not pass"

# ------------------------------------------------------------ warm the cache
# One real transcription, on synthetic audio, purely to make the engine compile
# itself now instead of the first time you use it. The transcript is garbage and
# is thrown away -- what matters is VLLM_CACHE_ROOT and FlashInfer's cache being
# populated for THIS gpu, since the compile cache is keyed by the card's model
# name and cannot be baked ahead of time for an unknown rental.
if [ "${MS_SKIP_WARM:-}" = "1" ]; then
  log "MS_SKIP_WARM=1 — leaving the engine cold"
else
  WARM="$MS_WORK/.warm"
  mkdir -p "$WARM"
  if [ ! -f "$WARM/warm.wav" ]; then
    # speech-shaped enough to produce windows; the content is irrelevant
    ffmpeg -v error -f lavfi -i "sine=frequency=200:duration=45" \
      -ac 1 -ar 16000 -c:a pcm_s16le "$WARM/warm.wav" \
      || log "!! could not make the warm-up clip; skipping"
  fi
  if [ -f "$WARM/warm.wav" ]; then
    log "warming the engine — this is the 240-400s you are paying so you do not have to later"
    t0=$(date +%s)
    ( cd "$WARM" && MS_WORK="$MS_WORK" HF_HOME="$HF_HOME" \
        "$MS_WORK/transcribe" "$WARM" >"$WARM/warm.log" 2>&1 ) || true
    # Judge this on whether the ENGINE came up, not on the exit code. A sine wave
    # contains no speech, so the run legitimately finds nothing and exits
    # nonzero -- while having done the entire job we wanted, which is to compile
    # and cache the engine for this GPU.
    if grep -q "engine up in" "$WARM/warm.log" 2>/dev/null; then
      log "engine warm after $(( $(date +%s) - t0 ))s ($(grep -o 'engine up in [0-9.]*s' "$WARM/warm.log" | head -1))"
    else
      log "!! the engine did not come up (see $WARM/warm.log) — the install is"
      log "   otherwise fine, but your first transcription will pay the compile"
    fi
    rm -f "$WARM"/*.txt "$WARM"/*.json "$WARM"/*.emb.npz
  fi
fi

# ------------------------------------------------------- the resident engine
# The warm cache above makes loading the engine ~70s instead of 240-400s. This
# makes it 0, which is a different problem: the cache is per MACHINE, the load
# is per PROCESS, so every ./transcribe paid it again. Keeping one engine alive
# in a daemon moves that cost to boot, once.
#
# It matters most for exactly the case the batch path cannot help -- a single
# recording, where 70s of loading sits in front of a few seconds of work.
#
# autostart, so the engine comes back when the INSTANCE does. Stopping and
# starting a vast instance keeps the disk and reruns the entrypoint, so the
# weights and caches survive but nothing is loaded -- without this the second
# boot silently costs 70s on every transcription, which is the whole problem
# this section exists to remove.
#
# Safe because of where this file is written: everything above had to succeed
# first, so this config only ever exists on a box that is fully installed. There
# is no half-provisioned boot for it to fail on.
MS_PY="$(. "$MS_WORK/env.sh" 2>/dev/null; echo "${MS_PY:-python3}")"
cat > /etc/supervisor/conf.d/meetscribe-engine.conf <<EOF
[program:meetscribe-engine]
command=$MS_PY -u $MS_WORK/pipeline/engined.py --serve
directory=$MS_WORK
autostart=true
autorestart=true
startsecs=45
startretries=10
stopwaitsecs=60
; The model runs in a separate VLLM::EngineCore process. Without these two,
; stopping or restarting this program signals only the parent and leaves that
; child alive holding the whole KV cache -- 17.6 GiB of a 24 GiB card -- which
; then makes the replacement fail to allocate. engined.py reaps such orphans at
; startup as a backstop, but not creating them is better.
stopasgroup=true
killasgroup=true
environment=MS_WORK="$MS_WORK",HF_HOME="$HF_HOME",VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT",OMP_NUM_THREADS="8",TOKENIZERS_PARALLELISM="false",VLLM_LOGGING_LEVEL="WARNING"
stdout_logfile=/var/log/portal/meetscribe-engine.log
redirect_stderr=true
EOF
supervisorctl reread >/dev/null 2>&1 || true
supervisorctl update >/dev/null 2>&1 || true

if [ "${MS_NO_DAEMON:-}" = "1" ]; then
  log "MS_NO_DAEMON=1 — not starting the resident engine"
else
  # Ask supervisor first, since that is what restarts the engine if it ever
  # dies. Then CHECK, and start it directly if that did nothing.
  #
  # Not a belt-and-braces flourish: supervisord was not running at all on a
  # provisioned vast box we tested, so `supervisorctl start` failed with "no
  # such file" and the engine simply never came up. supervisorctl fails quietly
  # enough that the difference between "started it" and "did nothing" is
  # invisible unless you look for the socket -- which is what this does.
  log "starting the resident engine"
  # `engine status` and NOT `[ -S run/engine.sock ]`. -S asks whether a socket
  # FILE exists, and a daemon that died hard leaves one behind -- so on the
  # no-supervisord box this whole section exists for, the wait below broke on its
  # first iteration against a dead socket, the direct-start fallback never fired,
  # and provisioning reported "resident engine up" over nothing at all. `status`
  # connects, which is the actual question.
  up() { MS_WORK="$MS_WORK" bash "$MS_WORK/engine" status >/dev/null 2>&1; }
  supervisorctl start meetscribe-engine >/dev/null 2>&1 || true
  for _ in $(seq 1 90); do up && break; sleep 2; done
  if ! up; then
    log "supervisor did not start it — starting it directly"
    MS_WORK="$MS_WORK" HF_HOME="$HF_HOME" VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
      bash "$MS_WORK/engine" start 2>&1 | sed 's/^/  /' || true
  fi
  # Everything works without it -- ./transcribe loads its own engine when there
  # is no daemon -- so this is a note, not a failure.
  if up; then
    log "resident engine up — transcriptions no longer pay the engine load"
  else
    log "!! the resident engine did not come up. See $MS_WORK/run/engine.log and"
    log "   /var/log/portal/meetscribe-engine.log. Transcribing still works;"
    log "   each run just loads its own engine, as it did before."
  fi
fi

# ---------------------------------------------------------------- done
date -u +%FT%TZ > "$MS_WORK/.provisioned"
log ""
log "ready. From your laptop:"
log "    ./transcribe ~/recordings/ --host <this-box>"
log "and for the browser UI, on the box:"
log "    $MS_WORK/ui &     then   ssh -N -L 8765:localhost:8765 <this-box>"
log "installed at $MS_WORK"
