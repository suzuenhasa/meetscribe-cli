#!/usr/bin/env bash
# Install the meetscribe pipeline.
#
#   ./setup.sh <sshhost>    copy this repo to a remote box and install there
#   ./setup.sh              on the machine that has the GPU: install here
#   ./setup.sh --check      verify an existing install, change nothing
#
# Idempotent. Everything lands inside this checkout, so deleting the directory
# removes the install. Re-run after a container recycle on an ephemeral box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Everything lives inside this checkout: the venv, the weights, the profile
# store, the work directories. Nothing is written outside it, so removing the
# directory removes the install, and two checkouts do not share state.
WORK="${MS_WORK:-$HERE}"
PIPE="$WORK/pipeline"
MODEL="OpenMOSS-Team/MOSS-Transcribe-Diarize"
WSP_REPO="Wespeaker/wespeaker-voxceleb-resnet34-LM"

# ---------------------------------------------------------------- pinned set
# Everything below floats otherwise: vLLM came from "latest release", both
# GitHub sources were shallow clones of whatever HEAD was that day, and both
# model downloads took the current revision. A clean install six months from now
# was therefore a materially different stack from the one every measurement in
# the README was taken on, and nothing recorded which.
#
# This is the set verified end-to-end on an RTX 3090, and the one today's
# benchmarks ran against. Captured from the working install, not guessed:
#
#   python 3.12.3   torch 2.13.0+cu130   torchaudio 2.11.0+cu130
#   transformers 5.15.0   numpy 2.3.5   scipy 1.18.0
#   huggingface-hub 1.27.0   tokenizers 0.22.2   soundfile 0.14.0
#
# MS_UNPINNED=1 takes current HEAD and latest instead, which is how you find out
# whether a newer stack works -- deliberately, rather than by the calendar.
VLLM_VER="${MS_VLLM_VER:-0.27.1}"
MOSS_SRC_REV="0e3d1403fd8f1f1c674e883ece96b9f630794ebe"
MOSS_MODEL_REV="e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
WSP_MODEL_REV="f0c48c298fd835726c27956a5d617bad7115627e"
WESPEAKER_SRC_REV="dfa741957e5c11f477623b6e583d67d0af25ee88"
if [ -n "${MS_UNPINNED:-}" ]; then
  VLLM_VER=""; MOSS_SRC_REV=""; MOSS_MODEL_REV=""; WSP_MODEL_REV=""; WESPEAKER_SRC_REV=""
fi
# Pin a git checkout to a commit a shallow clone did not fetch.
at_rev() {   # at_rev <dir> <sha>
  [ -n "$2" ] || return 0
  git -C "$1" rev-parse HEAD 2>/dev/null | grep -q "^$2" && return 0
  git -C "$1" fetch -q --depth 1 origin "$2" 2>/dev/null \
    && git -C "$1" checkout -q FETCH_HEAD 2>/dev/null \
    || warn "could not pin $(basename "$1") to ${2:0:12}; using $(git -C "$1" rev-parse --short HEAD 2>/dev/null)"
}
export HF_HOME="${HF_HOME:-$WORK/.hf_home}"

CHECK=0; REMOTE_HOST=""
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) REMOTE_HOST="$a" ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '   \033[31mXX\033[0m  %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------- drive a remote box
if [ -n "$REMOTE_HOST" ]; then
  say "Installing on $REMOTE_HOST"
  command -v rsync >/dev/null || die "rsync needed locally"
  ssh "$REMOTE_HOST" "mkdir -p '$WORK/_meetscribe_src'" \
    || die "cannot ssh to '$REMOTE_HOST' — check ~/.ssh/config"
  rsync -az --delete --exclude='.git/' --exclude='__pycache__/' \
    "$HERE/" "$REMOTE_HOST:$WORK/_meetscribe_src/"
  ok "sources copied"
  ssh -t "$REMOTE_HOST" "cd '$WORK/_meetscribe_src' && MS_WORK='$WORK' bash setup.sh $([ $CHECK -eq 1 ] && echo --check)"
  echo
  ok "done — now transcribe from HERE, not the box:"
  echo "       ./transcribe \"some meeting.mp3\"   (uses --host $REMOTE_HOST)"
  exit 0
fi

# ------------------------------------------------------------------- hardware
say "GPU"
command -v nvidia-smi >/dev/null || die "no nvidia-smi; this needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
[ -n "$CAP" ] && ok "compute capability $CAP$([ "${CAP%%.*}" -lt 8 ] 2>/dev/null && echo "  (pre-Ampere: float16 instead of bfloat16)")"
# vLLM's wheels target sm_70 and up. A GTX 1080 is sm_6.1 and fails deep inside
# vLLM with nothing that names the real cause, so say it plainly here instead.
if [ -n "$CAP" ] && [ "${CAP%%.*}" -lt 7 ] 2>/dev/null; then
  die "compute capability $CAP is too old. vLLM needs 7.0+ — RTX 20-series or
       newer. Pascal cards such as the GTX 1080 cannot run this."
fi
# The split between vLLM and the concurrent embedder is derived from the card at
# run time now, so there is nothing to pass by hand. Below ~7.5 GiB the embedder
# cannot sit alongside the engine, and transcription runs to completion first
# with the engine released before embedding -- slower, but complete.
if [ "$VRAM" -lt 6000 ]; then
  warn "${VRAM} MiB VRAM is under the floor. The engine needs ~2.6 GiB plus KV"
  warn "cache, and the speaker embedder ~1.5 more. It will refuse to start rather"
  warn "than fail after loading the weights."
elif [ "$VRAM" -lt 7500 ]; then
  warn "${VRAM} MiB VRAM — transcription and embedding will run in two passes"
  warn "rather than overlapping. Measured ~19x realtime on a 6 GB RTX 2060."
fi
command -v ffmpeg >/dev/null || die "ffmpeg missing (apt-get install -y ffmpeg)"
ok "ffmpeg present"

# ---------------------------------------------------------------- interpreter
# Use whichever python ALREADY owns torch. Installing into a different env makes
# pip pull a second torch plus the whole CUDA stack, which then shadows the
# working one and everything fails in confusing ways.
say "Python"
mkdir -p "$WORK"
# Reuse an interpreter ONLY if it already has torch AND vLLM -- that means a
# prebuilt GPU image, where installing alongside is right. An interpreter with
# torch but no vLLM is usually a conda base or a system python; installing vLLM
# there drags in its own torch, can break the environment it was borrowed from,
# and is not ours to modify. Build a venv inside the checkout instead.
PY=""; MODE=""
for c in "${MS_PY:-}" "$WORK/venv/bin/python" /venv/main/bin/python /usr/bin/python3 python3; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import torch, vllm" >/dev/null 2>&1; then PY="$c"; MODE=both; break; fi
done
if [ -z "$PY" ] && [ -x "$WORK/venv/bin/python" ]; then
  PY="$WORK/venv/bin/python"; MODE=fresh      # half-built venv from an interrupted run
fi
if [ -z "$PY" ]; then
  [ "$CHECK" -eq 1 ] && die "no interpreter with vLLM"
  warn "no interpreter with vLLM — building one at $WORK/venv (vLLM brings its own torch)"
  if [ ! -x "$WORK/venv/bin/python" ]; then
    if command -v uv >/dev/null; then
      uv venv "$WORK/venv" --python 3.12 >/dev/null \
        || die "uv could not create a venv at $WORK/venv"
    else
      # uv is faster but not required. The stdlib module is always there, so a
      # bare machine should not be told to go and install a package manager
      # first just to get started.
      python3 -m venv "$WORK/venv" \
        || die "python3 -m venv failed — on Debian/Ubuntu: apt-get install -y python3-venv"
      "$WORK/venv/bin/python" -m pip install -q -U pip setuptools wheel \
        || die "could not bootstrap pip inside $WORK/venv"
    fi
  fi
  PY="$WORK/venv/bin/python"; MODE=fresh
  ok "created $WORK/venv"
fi
# CUDA wheel selection. torch on PyPI ships one default CUDA build, currently
# cu130, and a driver older than 13.x cannot load it -- the failure is
# "CUDA initialization: The NVIDIA driver on your system is too old", after
# several GB have already downloaded. CUDA 12.x wheels run on any 12.x driver
# (minor-version compatibility), so match the wheel to the driver.
DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
if command -v uv >/dev/null; then
  # uv resolves the right CUDA variant from the installed driver itself.
  PIPI=(uv pip install --python "$PY" --torch-backend=auto)
  [ -n "$DRV_CUDA" ] && ok "driver supports CUDA $DRV_CUDA — uv will match the wheels"
else
  PIPI=("$PY" -m pip install)
  case "$DRV_CUDA" in
    13.*|1[4-9].*) : ;;                                   # default PyPI build is fine
    12.[89]|12.1[0-9]) PIPI+=(--extra-index-url https://download.pytorch.org/whl/cu128) ;;
    12.*)              PIPI+=(--extra-index-url https://download.pytorch.org/whl/cu126) ;;
    *) warn "could not read the driver's CUDA version; using default wheels" ;;
  esac
  [ -n "$DRV_CUDA" ] && ok "driver supports CUDA $DRV_CUDA"
fi

# vLLM before the torch checks: a freshly created venv has neither, and vLLM
# brings a matching torch with it.
#
# The wheel on PyPI is built against CUDA 13, whose runtime a 12.x driver cannot
# load -- torch installs fine and then `import vllm` dies on
# "libcudart.so.13: cannot open shared object file". vLLM also publishes a +cu129
# wheel per release, and CUDA 12.x has minor-version compatibility, so that one
# runs on any 12.x driver. Pick by the driver.
if ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "vLLM not installed in $PY"
  warn "installing vLLM (~2-3 GB, several minutes — progress below)"
  VLLM_SPEC="${VLLM_VER:+vllm==$VLLM_VER}"; VLLM_SPEC="${VLLM_SPEC:-vllm}"
  case "$DRV_CUDA" in
    12.*)
      # The pinned version, not whatever is newest today.
      VER="$VLLM_VER"
      [ -n "$VER" ] || VER="$(curl -fsSL https://api.github.com/repos/vllm-project/vllm/releases/latest 2>/dev/null \
             | grep -m1 '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')"
      ARCH="$(uname -m)"
      if [ -n "$VER" ]; then
        CAND="https://github.com/vllm-project/vllm/releases/download/v${VER}/vllm-${VER}+cu129-cp38-abi3-manylinux_2_28_${ARCH}.whl"
        if curl -fsI "$CAND" >/dev/null 2>&1; then
          VLLM_SPEC="$CAND"
          ok "driver is CUDA $DRV_CUDA — using the cu129 build of vLLM $VER"
        else
          warn "no cu129 wheel published for vLLM $VER; trying the default build"
        fi
      fi
      ;;
  esac
  "${PIPI[@]}" "$VLLM_SPEC" || die "vLLM install failed. Output is above; the usual causes are
       disk space, and a CUDA/torch combination no published wheel matches."
  "$PY" -c "import vllm" >/dev/null 2>&1 || die "vLLM installed but will not import into
       $PY. If the error mentions libcudart.so.13, the wheel wants CUDA 13 and this
       driver is CUDA $DRV_CUDA — either update the NVIDIA driver, or install the
       +cu129 wheel for your vLLM version from the project's GitHub releases."
fi

TORCH_BEFORE=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null) \
  || die "no torch in $PY after installing vLLM"
ok "$PY"
ok "torch $TORCH_BEFORE"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch cannot see the GPU from $PY"
ok "CUDA visible"

# --------------------------------------------------------------- pid headroom
# vLLM's engine spawns processes; a tight cgroup pid cap kills it with an opaque
# "Engine core initialization failed".
if [ -r /sys/fs/cgroup/pids.max ] && [ -r /sys/fs/cgroup/pids.current ]; then
  PMAX=$(cat /sys/fs/cgroup/pids.max); PCUR=$(cat /sys/fs/cgroup/pids.current)
  if [ "$PMAX" != max ] && [ $((PMAX - PCUR)) -lt 300 ]; then
    warn "only $((PMAX - PCUR)) pids free — vLLM may fail to start"
    if [ "$CHECK" -eq 0 ] && command -v supervisorctl >/dev/null \
       && supervisorctl status ray 2>/dev/null | grep -q RUNNING; then
      warn "stopping the idle Ray cluster (undo: supervisorctl start ray)"
      supervisorctl stop ray >/dev/null || true
    fi
  fi
fi

# ---------------------------------------------------------------------- deps
say "Dependencies"
"$PY" - <<'EOF' || die "this vLLM build lacks MOSS support; upgrade vLLM"
from vllm import ModelRegistry; import sys
sys.exit(0 if "MossTranscribeDiarizeForConditionalGeneration" in ModelRegistry.get_supported_archs() else 1)
EOF
ok "vLLM $("$PY" -c 'import vllm;print(vllm.__version__)') with MOSS in-tree"

if ! "$PY" -c "import moss_transcribe_diarize" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "moss_transcribe_diarize missing"
  [ -d "$WORK/MOSS-Transcribe-Diarize" ] || git clone -q --depth 1 \
      https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git "$WORK/MOSS-Transcribe-Diarize"
  at_rev "$WORK/MOSS-Transcribe-Diarize" "$MOSS_SRC_REV"
  "${PIPI[@]}" -e "$WORK/MOSS-Transcribe-Diarize"
fi
ok "moss_transcribe_diarize"

MISSING=()
for m in scipy soundfile huggingface_hub torchaudio yaml numpy; do
  "$PY" -c "import $m" >/dev/null 2>&1 || MISSING+=("$( [ "$m" = yaml ] && echo pyyaml || echo "$m")")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  [ "$CHECK" -eq 1 ] && die "missing: ${MISSING[*]}"
  "${PIPI[@]}" "${MISSING[@]}"
fi
ok "scipy, soundfile, torchaudio, pyyaml, huggingface_hub"

TORCH_AFTER=$("$PY" -c "import torch; print(torch.__version__)")
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || die "a dependency moved torch $TORCH_BEFORE -> $TORCH_AFTER"
ok "torch unchanged"

# ------------------------------------------------------------------- weights
say "Models (~2 GB on a cold box — this is the slow part)"
# inbox/ and library/ are the two directories a person actually uses: drop
# recordings in one, find meetings in the other. They have to exist before the
# first run, since ./transcribe with no argument means the inbox and the README
# tells you to copy into it.
mkdir -p "$HF_HOME" "$WORK/runs" "$WORK/out" "$WORK/inbox" "$WORK/library" \
         "$WORK/wsp_ckpt/resnet34"
if [ "$CHECK" -eq 1 ]; then
  "$PY" -c "
from huggingface_hub import snapshot_download as d; d('$MODEL', local_files_only=True, revision='$MOSS_MODEL_REV' or None)" >/dev/null 2>&1 \
    && ok "MOSS cached" || die "MOSS weights not downloaded"
else
  # Show the progress bars. Sending this to /dev/null made setup sit silent for
  # several minutes on a cold box, which reads as "it is not downloading the
  # weights" -- and then the first run looks like it fetched them instead.
  if "$PY" -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', revision='$MOSS_MODEL_REV' or None)
print(p)
" ; then
    ok "MOSS-Transcribe-Diarize"
  else
    die "MOSS download failed (network? disk? HF_HOME=$HF_HOME)"
  fi
  # Prove it landed, rather than trusting the exit code.
  "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_files_only=True, revision='$MOSS_MODEL_REV' or None)" >/dev/null 2>&1 \
    || die "MOSS reported success but is not in the cache at $HF_HOME"
fi

if [ ! -f "$WORK/wsp_ckpt/resnet34/avg_model.pt" ]; then
  [ "$CHECK" -eq 1 ] && die "WeSpeaker checkpoint missing"
  warn "downloading WeSpeaker ResNet34-LM (~43 MB)"
  "$PY" - <<EOF || die "WeSpeaker download failed"
from huggingface_hub import hf_hub_download
import shutil
# this repo names the weights `avg_model`, without the extension the
# ResNet293 one used; the local name stays avg_model.pt either way.
for remote, local in (("avg_model", "avg_model.pt"), ("config.yaml", "config.yaml")):
    shutil.copy(hf_hub_download("$WSP_REPO", remote, revision="$WSP_MODEL_REV" or None),
                "$WORK/wsp_ckpt/resnet34/" + local)
EOF
  [ -s "$WORK/wsp_ckpt/resnet34/avg_model.pt" ] || die "WeSpeaker checkpoint is empty"
fi
ok "WeSpeaker ResNet34-LM ($(du -h "$WORK/wsp_ckpt/resnet34/avg_model.pt" | cut -f1))"

if [ ! -d "$WORK/wespeaker_src/wespeaker" ]; then
  [ "$CHECK" -eq 1 ] && die "wespeaker source missing"
  git clone -q --depth 1 https://github.com/wenet-e2e/wespeaker.git "$WORK/wespeaker_src"
  at_rev "$WORK/wespeaker_src" "$WESPEAKER_SRC_REV"
fi
ok "wespeaker source"

# Import the embedder for real. wespeaker/__init__.py pulls in its CLI, which
# wants silero_vad; the embedder sidesteps that, but a checkpoint or torch
# mismatch still only shows up at import time. Catching it here beats discovering
# it after a batch has already spent minutes transcribing.
MS_WORK="$WORK" "$PY" -c "
import os, sys, importlib.util as u
spec = u.spec_from_file_location('eb', '$PIPE/link/embed_batched.py')
m = u.module_from_spec(spec)
sys.argv = ['x']
try:
    spec.loader.exec_module(m)
except SystemExit:
    pass
" 2>/dev/null || die "embed_batched.py cannot import — see $PIPE/link/embed_batched.py"
ok "embedder imports"

# ------------------------------------------------------------------- scripts
say "Pipeline"
if [ "$CHECK" -eq 1 ]; then
  for f in transcribe_meeting.py batch.py cluster_speakers.py mktxt.py speakers.py \
           identify.py link/link.py link/embed_batched.py; do
    [ -f "$PIPE/$f" ] || die "missing $PIPE/$f"
  done
  ok "scripts in place"
  [ -f "$WORK/speakers.db" ] || die "speaker database missing ($WORK/speakers.db)"
  ok "speaker database ($("$PY" -c "
import sqlite3;print(sqlite3.connect('$WORK/speakers.db').execute('select count(*) from speakers').fetchone()[0])" 2>/dev/null || echo 0) voices enrolled)"
else
  cat > "$WORK/env.sh" <<EOF
# source $WORK/env.sh
export MS_WORK="$WORK"
export MS_PIPE="$PIPE"
export HF_HOME="$HF_HOME"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export MS_PY="$PY"
EOF
  ok "wrote $WORK/env.sh"
  # So `./transcribe --host thisbox` from another machine can find the install
  # without being told where it is. Best effort: it needs root, and not having
  # it only means the caller has to pass MS_REMOTE.
  echo "$WORK" > /etc/meetscribe-work 2>/dev/null \
    && ok "recorded the install path in /etc/meetscribe-work" || true

  # The profile store. Created empty and left in place on re-runs: it is the one
  # piece of state here that cannot be rebuilt from the audio, because it holds
  # the names a person typed.
  MS_WORK="$WORK" "$PY" -c "
import os, sys
sys.path.insert(0, '$PIPE')
import speakers
c = speakers.db(); c.commit()
n = c.execute('select count(*) from speakers').fetchone()[0]
print('   \033[32mok\033[0m  speaker database %s (%d voice%s enrolled)'
      % (speakers.DB, n, '' if n == 1 else 's'))"
fi

say "Ready"
cat <<EOF
   Drop recordings in inbox/ and run it:

     cp ~/recordings/*.mp3 inbox/
     ./transcribe

   Each becomes a directory in library/ — the transcript, the audio, and a few
   seconds of each voice. The inbox empties as they finish, so anything still
   sitting there has not been done.

     ./speakers meetings                       what is in the library
     ./speakers who <meeting>                  the voices in it
     ./speakers name <meeting> G02 "Bob"       remember one
     ./speakers apply --apply                  name them in older meetings too

   For names the model has never heard, put a glossary.txt next to where you
   run it (one term per line) — see glossary.txt.example.

   Every command, in full:  RUNBOOK.md
   Verify this install any time, changing nothing:
     ./setup.sh --check
EOF
