# Running this on vast.ai

A template so a rented box comes up with meetscribe installed and the engine
already compiled, instead of you doing it again every time.

## The template

In the vast console, **Templates → New**:

| field | value |
|---|---|
| Image | `vastai/vllm:v0.27.1-cuda-12.9` |
| Launch mode | **SSH**, direct SSH on |
| On-start | `exec /opt/instance-tools/bin/entrypoint.sh` |
| Ports | none |
| Disk | 60 GB |

Environment:

```
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/suzuenhasa/meetscribe-cli/<COMMIT>/vast/provision.sh
MS_WORK=/opt/meetscribe
MS_REF=<COMMIT>
HF_HOME=/opt/meetscribe/.hf_home
VLLM_CACHE_ROOT=/opt/meetscribe/.vllm_cache
```

Four of those are load-bearing in ways that are not obvious:

**The on-start line is not optional.** SSH mode replaces the image's entrypoint,
so without it supervisord never starts — and it fails *silently*, giving you an
instance that looks fine and has provisioned nothing.

**Pin `PROVISIONING_SCRIPT` to a commit, not `main`.** It is fetched over HTTPS
and run as root, so whoever controls that URL's contents controls every instance
you launch. A tag is not enough; tags move.

Use the **full 40-character** sha in both places. `git fetch` resolves a name on
the remote, and a shortened sha is not one — it fails with `couldn't find remote
ref`. The script checks for this and says so rather than letting you find out
from git.

**No ports.** SSH mode maps 22 and that is the whole interface — `./transcribe`
works over ssh and rsync. The web UI has no authentication of any kind, and a
vast box has a public IP, so it is reached through an ssh tunnel rather than
published. See below.

**`MS_WORK` is not `/workspace`.** Vast documents `/workspace` as possibly shared
between instances with concurrent writers. `speakers.db` is the one file here
that cannot be rebuilt from the audio.

Optionally filter offers to machines that can actually run it: compute
capability ≥ 7.0, and ≥ 8 GB of VRAM *per GPU* (that filter is per-GPU, not the
total across a multi-GPU box).

## What provisioning does

Clones the repo, runs `setup.sh`, verifies with `setup.sh --check`, and then
transcribes 45 seconds of synthetic audio and throws the result away.

That last step is the point. The **first** engine load on any machine costs
240–400 s while torch.compile and FlashInfer populate their caches; every load
after is ~70 s. Doing it during provisioning means the box is warm before you
touch it. Set `MS_SKIP_WARM=1` to skip it.

It cannot be baked into an image ahead of time: vLLM's compile cache is keyed by
the GPU's *model name string*, so a cache built on a 5090 misses on a 5080, and
you do not know what you have rented until you have rented it.

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
```

If `.provisioned` is missing, provisioning did not finish — the log says why.
Weights and the compile cache live under `MS_WORK`, so re-running provisioning
after a failure picks up where it left off rather than starting again.
