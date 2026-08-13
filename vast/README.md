# Running this on vast.ai

A template so a rented box comes up with meetscribe installed and the engine
already compiled, instead of you doing it again every time.

## The template

In the vast console, **Templates → New**:

| field | value |
|---|---|
| Image | `vastai/vllm:v0.27.1-cuda-12.9` |
| Launch mode | **SSH**, direct SSH on |
| On-start Script | paste all of `vast/provision.sh` |
| Ports | none |
| Disk | 60 GB |

That is the whole setup. Paste the script, set nothing else.

To pin a version — and you should, because the default tracks `main` and `main`
moves — edit the one line near the top of what you pasted:

```bash
MS_REF="${MS_REF:-main}"      ->      MS_REF="${MS_REF:-<full 40-char sha>}"
```

Use the **full 40 characters**. `git fetch` resolves a ref name on the remote and
an abbreviated sha is not one, so it fails with `couldn't find remote ref`. The
script checks for this and says so rather than letting you find out from git.

**No ports.** SSH mode maps 22 and that is the whole interface — `./transcribe`
works over ssh and rsync. The web UI has no authentication of any kind, and a
vast box has a public IP, so it is reached through an ssh tunnel rather than
published. See below.

### The other way in

vast also fetches a script from a URL, which is worth using if you would rather
not paste a few hundred lines into a form, or want several templates sharing one
script. Set these two instead of the On-start paste, and put
`exec /opt/instance-tools/bin/entrypoint.sh` in On-start:

```
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/suzuenhasa/meetscribe-cli/<sha>/vast/provision.sh
MS_REF=<the same sha>
```

Pin it to a commit rather than a branch. It is fetched over HTTPS and run as
root, so whoever controls that URL's contents controls every instance you
launch, and a tag is not enough because tags move.

Either way, `MS_WORK`, `HF_HOME` and `VLLM_CACHE_ROOT` are already the script's
defaults (`/opt/meetscribe` and two directories under it) and do not need
setting. `MS_WORK` is deliberately not `/workspace`: vast documents that as
possibly shared between instances with concurrent writers, and `speakers.db` is
the one file here that cannot be rebuilt from the audio.

Optionally filter offers to machines that can actually run it: compute
capability >= 7.0, and >= 8 GB of VRAM *per GPU* (that filter is per-GPU, not the
total across a multi-GPU box).

## What provisioning does

Clones the repo, runs `setup.sh`, verifies with `setup.sh --check`, transcribes
45 seconds of synthetic audio and throws the result away, then starts the
engine and leaves it running.

The last two steps are the point, and they solve two different halves of the
same cost.

**The throwaway transcription** fills the compile caches. The *first* engine
load on any machine costs 240–400 s while torch.compile and FlashInfer populate
them; every load after is ~70 s. This cannot be baked into an image ahead of
time — vLLM's compile cache is keyed by the GPU's *model name string*, so a
cache built on a 5090 misses on a 5080, and you do not know what you have rented
until you have rented it. `MS_SKIP_WARM=1` skips it.

**The resident engine** removes the remaining ~70 s. That load is paid per run,
not per machine, so without a daemon every `./transcribe` pays it again — which
on a short recording is the entire wall clock. Measured on a 3090: a 3-minute
clip took 145 s cold and 25 s against a resident engine. `MS_NO_DAEMON=1` skips
it, and `./engine start|stop|status` controls it by hand afterwards.

Neither is load-bearing. Without both, transcribing works exactly as it always
did; it is just slower to start.

## Using it

```bash
# from your laptop, audio stays local, only the GPU work goes over
./transcribe ~/recordings/ --host <box>

# the browser UI, on the box, reached through a tunnel
ssh <box> '/opt/meetscribe/ui /opt/meetscribe/library &'
ssh -N -L 8765:localhost:8765 <box>
# then http://localhost:8765
```

Do **not** bind the UI to `0.0.0.0` to skip the tunnel. It has no login, and the
library it serves is your meetings.

## Checking it worked

```bash
cat /var/log/portal/provisioning.log      # what provisioning did
cat /opt/meetscribe/.provisioned          # the timestamp it finished
/opt/meetscribe/setup.sh --check          # verify, changes nothing
/opt/meetscribe/engine status             # is the engine resident?
```

Do check that last one rather than assuming. The engine is registered with
supervisor so it restarts if it dies — but supervisord was not running at all on
one provisioned box we tested, and `supervisorctl start` fails quietly enough
that nothing looked wrong. Provisioning now starts the engine directly when that
happens, and `./engine start` does the same by hand.

If `.provisioned` is missing, provisioning did not finish — the log says why.
Weights and the compile cache live under `MS_WORK`, so re-running provisioning
after a failure picks up where it left off rather than starting again.
