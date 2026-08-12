#!/usr/bin/env bash
# Install the meetscribe pipeline.
#
#   ./setup.sh msbox        from your laptop: copy this repo to the box and install there
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
WSP_REPO="Wespeaker/wespeaker-voxceleb-resnet293-LM"
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
if [ "$VRAM" -lt 9000 ]; then
  warn "${VRAM} MiB VRAM is tight. Weights are ~1.7 GiB and overhead ~0.7 GiB, so it"
  warn "fits, but pass --gpu-frac 0.80 for a single file and 0.65 for a folder, or"
  warn "the concurrent embedder will not have room."
elif [ "$VRAM" -lt 12000 ]; then
  warn "only ${VRAM} MiB VRAM — usable; drop --gpu-frac if the embedder OOMs"
fi
command -v ffmpeg >/dev/null || die "ffmpeg missing (apt-get install -y ffmpeg)"
ok "ffmpeg present"

# ---------------------------------------------------------------- interpreter
# Use whichever python ALREADY owns torch. Installing into a different env makes
# pip pull a second torch plus the whole CUDA stack, which then shadows the
# working one and everything fails in confusing ways.
say "Python"
mkdir -p "$WORK"
PY=""; MODE=""
for c in "${MS_PY:-}" /usr/bin/python3 /venv/main/bin/python "$WORK/venv/bin/python" python3; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import torch, vllm" >/dev/null 2>&1; then PY="$c"; MODE=both; break; fi
  [ -z "$PY" ] && "$c" -c "import torch" >/dev/null 2>&1 && { PY="$c"; MODE=torch; }
done
if [ -z "$PY" ]; then
  [ "$CHECK" -eq 1 ] && die "no interpreter with torch"
  warn "no torch anywhere — building a venv at $WORK/venv (vLLM brings its own torch)"
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
if command -v uv >/dev/null; then PIPI=(uv pip install --python "$PY" -q); else PIPI=("$PY" -m pip install -q); fi
[ "$MODE" = fresh ] && { say "Installing vLLM (~2-3 GB, several minutes)"; "${PIPI[@]}" vllm || die "vLLM install failed"; }
TORCH_BEFORE=$("$PY" -c "import torch; print(torch.__version__)")
ok "$PY  torch $TORCH_BEFORE"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "torch cannot see the GPU"
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
if ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "vLLM not installed"
  "${PIPI[@]}" vllm
fi
"$PY" - <<'EOF' || die "this vLLM build lacks MOSS support; upgrade vLLM"
from vllm import ModelRegistry; import sys
sys.exit(0 if "MossTranscribeDiarizeForConditionalGeneration" in ModelRegistry.get_supported_archs() else 1)
EOF
ok "vLLM $("$PY" -c 'import vllm;print(vllm.__version__)') with MOSS in-tree"

if ! "$PY" -c "import moss_transcribe_diarize" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "moss_transcribe_diarize missing"
  [ -d "$WORK/MOSS-Transcribe-Diarize" ] || git clone -q --depth 1 \
      https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git "$WORK/MOSS-Transcribe-Diarize"
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
mkdir -p "$HF_HOME" "$WORK/runs" "$WORK/out" "$WORK/inbox" "$WORK/wsp_ckpt/resnet293"
if [ "$CHECK" -eq 1 ]; then
  "$PY" -c "
from huggingface_hub import snapshot_download as d; d('$MODEL', local_files_only=True)" >/dev/null 2>&1 \
    && ok "MOSS cached" || die "MOSS weights not downloaded"
else
  # Show the progress bars. Sending this to /dev/null made setup sit silent for
  # several minutes on a cold box, which reads as "it is not downloading the
  # weights" -- and then the first run looks like it fetched them instead.
  if "$PY" -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL')
print(p)
" ; then
    ok "MOSS-Transcribe-Diarize"
  else
    die "MOSS download failed (network? disk? HF_HOME=$HF_HOME)"
  fi
  # Prove it landed, rather than trusting the exit code.
  "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_files_only=True)" >/dev/null 2>&1 \
    || die "MOSS reported success but is not in the cache at $HF_HOME"
fi

if [ ! -f "$WORK/wsp_ckpt/resnet293/avg_model.pt" ]; then
  [ "$CHECK" -eq 1 ] && die "WeSpeaker checkpoint missing"
  warn "downloading WeSpeaker ResNet293-LM (~134 MB)"
  "$PY" - <<EOF || die "WeSpeaker download failed"
from huggingface_hub import hf_hub_download
import shutil
for f in ("avg_model.pt", "config.yaml"):
    shutil.copy(hf_hub_download("$WSP_REPO", f), "$WORK/wsp_ckpt/resnet293/" + f)
EOF
  [ -s "$WORK/wsp_ckpt/resnet293/avg_model.pt" ] || die "WeSpeaker checkpoint is empty"
fi
ok "WeSpeaker ResNet293-LM ($(du -h "$WORK/wsp_ckpt/resnet293/avg_model.pt" | cut -f1))"

if [ ! -d "$WORK/wespeaker_src/wespeaker" ]; then
  [ "$CHECK" -eq 1 ] && die "wespeaker source missing"
  git clone -q --depth 1 https://github.com/wenet-e2e/wespeaker.git "$WORK/wespeaker_src"
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
   Run this from the machine your AUDIO is on:

     ./transcribe "some meeting.mp3"

   It uploads, transcribes, identifies who spoke, and writes the transcript
   into your current directory. No flags needed.

   For names the model has never heard, put a glossary.txt next to where you
   run it (one term per line) — see glossary.txt.example.

   Verify this install any time, changing nothing:
     ./setup.sh --check
EOF
