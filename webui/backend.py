"""Where the GPU work happens: it runs ./transcribe.

This used to reimplement the pipeline's own upload -> run -> retrieve logic, in
two classes, one for a local install and one for ssh. All of that already exists
in ./transcribe, which also carries things this file never had -- the embedding
retry for when the engine will not share the card, cross-file window pooling, the
worker pool for post-processing, and exit codes that distinguish "some meetings
failed" from "the run failed". Duplicating it meant the UI ran an older, worse
pipeline than the CLI did, and every fix had to be made twice.

So there is one path now. ./transcribe decides local vs remote itself (--host, or
MS_HOST), so this does not need to know which it is: only transcription and
embedding need a GPU, and everything else -- clustering, identification, renaming
-- runs wherever the server runs.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # the meetscribe-cli checkout
TRANSCRIBE = ROOT / "transcribe"

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".webm")


def _stage(files, into: Path):
    """Link the batch into one directory, because ./transcribe takes one path.

    Hard link where the filesystem allows it, symlink otherwise, so a library of
    hour-long recordings is not copied just to start a run -- transcribe finds
    both, it uses find -L. Colliding basenames get a numeric suffix rather than
    silently clobbering, since two folders can hold the same name.
    """
    into.mkdir(parents=True, exist_ok=True)
    staged, seen = [], {}
    for f in files:
        src = Path(f).resolve()
        n = seen.get(src.name, 0)
        seen[src.name] = n + 1
        dst = into / (src.name if n == 0 else f"{src.stem}-{n}{src.suffix}")
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src, dst)
        staged.append((src, dst))
    return staged


class Backend:
    """Runs ./transcribe. Local or over ssh is transcribe's decision, not ours."""

    def __init__(self, host="", work=None):
        self.host = host or ""
        self.work = Path(work) if work else ROOT
        self.name = f"remote ({self.host})" if self.host else "local"

    def available(self):
        """Is there anything to run -- a local install, or a reachable box?"""
        if (self.work / "pipeline" / "transcribe_meeting.py").exists():
            return True
        if not self.host:
            return False
        try:
            return subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", self.host,
                 f"test -f {self.work}/pipeline/transcribe_meeting.py"],
                capture_output=True, timeout=20).returncode == 0
        except Exception:
            return False

    def run_batch(self, files, out_dir: Path, glossary="", roster="", on_stage=None):
        """Transcribe everything. -> [(source_path, linked_json_path), ...]

        Only meetings that actually produced a transcript are returned. A nonzero
        exit means SOME recording did not complete, not that the run failed --
        the others are on disk and still worth having, so this reports per file
        rather than raising. Treating a partial batch as a total loss is the bug
        that used to throw away a whole good run over one bad file.
        """
        files = list(files)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = out_dir / "_batch.log"
        if on_stage:
            on_stage("transcribe", 5)

        with tempfile.TemporaryDirectory(prefix="ms-batch-") as tmp:
            audio_dir = Path(tmp) / "audio"
            staged = _stage(files, audio_dir)
            argv = [str(TRANSCRIBE), str(audio_dir)]
            if glossary:
                argv += ["--glossary", glossary]
            if roster:
                argv += ["--roster", roster]
            if self.host:
                argv += ["--host", self.host]
            with open(log, "a") as fh:
                fh.write("\n$ " + " ".join(argv) + "\n")
                fh.flush()
                # transcribe writes its transcripts into the working directory
                rc = subprocess.run(
                    argv, cwd=out_dir, stdout=fh, stderr=subprocess.STDOUT,
                    env={**os.environ, "MS_WORK": str(self.work)}).returncode

            out = []
            for src, staged_path in staged:
                js = out_dir / f"{staged_path.stem}.json"
                txt = out_dir / f"{staged_path.stem}.txt"
                if not js.exists() or js.stat().st_size == 0:
                    continue
                # put the caller's own name back on the meeting
                final_js, final_txt = out_dir / f"{src.stem}.json", out_dir / f"{src.stem}.txt"
                if js != final_js:
                    shutil.move(str(js), final_js)
                    if txt.exists():
                        shutil.move(str(txt), final_txt)
                out.append((src, final_js))

        if on_stage:
            on_stage("done", 100)
        if rc != 0 or len(out) < len(files):
            print(f"[backend] {len(files) - len(out)} of {len(files)} recording(s) "
                  f"did not complete — see {log}", flush=True)
        return out

    def run_pipeline(self, audio, out_dir: Path, glossary="", on_stage=None):
        got = self.run_batch([audio], out_dir, glossary, "", on_stage)
        return got[0][1] if got else None


def from_config(cfg):
    """cfg: {"host": "sshhost" or "", "work": path}. No host means local."""
    return Backend(cfg.get("host", ""), cfg.get("work"))
