# Omnilingual-ASR Server

A community-maintained FastAPI wrapper for running
[Meta's Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
behind a small, OpenAI-style transcription API.

This repository and its container images are not official Meta projects. They
do not contain training data. A build downloads the selected model weights
from the locations configured by the upstream Omnilingual ASR/fairseq2 asset
cards.

## Status

| Path | Status | Notes |
| --- | --- | --- |
| CPU, Python 3.12 | Validated | Model load and real transcription tested with the default CTC 300M v2 model |
| CUDA 12.6 | Experimental | Dependency lock resolves; the updated image has not completed a GPU runtime test |
| OpenAI compatibility | Partial | File transcription, model listing, JSON, and plain-text responses only |

The default build is CPU-only and contains no NVIDIA Python packages. The
optional CUDA target is for Linux/amd64 hosts with an NVIDIA driver compatible
with CUDA 12.6.

## Quick start with Docker

Requirements: Docker with BuildKit, a Linux/amd64 runtime (or compatible
emulation), internet access during the build, and several gigabytes of disk.
The default checkpoint alone is about 1.3 GiB according to the upstream model
table; Python, PyTorch, build layers, and caches need additional space.

Build the CPU image. The model is downloaded once during this step:

```bash
docker build --target default -t omniasr-server:cpu .
```

Run it on localhost:

```bash
docker run --rm \
  --publish 127.0.0.1:8080:8080 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  omniasr-server:cpu
```

The image runs as numeric user/group `65532:65532`, includes a health check,
and has the selected model preloaded. Wait for `Model ... loaded successfully`
or for the container to report healthy before sending audio.

Check the service:

```bash
curl http://127.0.0.1:8080/health-check
curl http://127.0.0.1:8080/v1/models
```

Transcribe a local WAV or FLAC file:

```bash
curl --fail-with-body \
  --request POST http://127.0.0.1:8080/v1/audio/transcriptions \
  --form file=@audio.wav \
  --form response_format=json
```

Example response:

```json
{"text":"transcribed speech"}
```

Use `--form response_format=text` for an unwrapped text response. Audio bytes
are processed locally by the container; this wrapper does not accept media
URLs or upload audio to an external transcription API.

## Local Python setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python main.py
```

The first local start downloads the configured model to fairseq2's cache. Run
the tests without loading model weights:

```bash
uv run --frozen --extra cpu pytest -q
```

The dependency rationale and safe upgrade procedure are in
[DEPENDENCIES.md](DEPENDENCIES.md), including the documented upstream
`transformers` advisory constraint and its operating mitigations.

## Client examples

### curl

```bash
curl --fail-with-body \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  --form file=@audio.flac \
  --form model=omniASR_CTC_300M_v2
```

The endpoint requires a multipart file upload. WAV and FLAC are the safest
choices; other formats depend on the codecs available through libsndfile.

### Python OpenAI client

The example is a standalone uv script and installs its exact client version in
uv's script environment:

```bash
uv run scripts/openai_client.py ./audio.wav
```

For an LLM model variant, an optional language hint can be supplied:

```bash
uv run scripts/openai_client.py ./audio.wav --language xho_Latn
```

CTC and W2V variants ignore `language`. LLM variants accept Omnilingual ASR
language-script identifiers; ISO 639-1 values are mapped heuristically in
[`app/languages.py`](app/languages.py).

## API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health-check` | Process readiness after startup/model load |
| `GET` | `/v1/models` | The single configured model |
| `POST` | `/v1/audio/transcriptions` | Transcribe one uploaded audio file |

Supported transcription fields:

| Field | Behavior |
| --- | --- |
| `file` | Required multipart upload |
| `language` | Used only by LLM model variants |
| `response_format` | `json` (default) or `text` |
| `model` | Accepted for client compatibility; the loaded server model is used |
| `prompt`, `temperature`, `timestamp_granularities` | Accepted but currently ignored |

Formats such as `verbose_json`, `srt`, and `vtt` are not implemented and
return an OpenAI-shaped HTTP 400 error instead of silently returning the wrong
format. Errors use this shape:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "file",
    "code": null
  }
}
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | `omniASR_CTC_300M_v2` | Model asset card loaded at startup |
| `OMNILINGUAL_HOST` | `0.0.0.0` | Listen address inside the container |
| `OMNILINGUAL_PORT` | `8080` | Listen port |
| `OMNILINGUAL_DTYPE` | `auto` | `auto`, `float16`, `bfloat16`, or `float32` |

CPU always defaults to FP32. CUDA `auto` uses BF16 on devices with compute
capability 8.0 or newer and FP16 on older devices. On a GTX 1660 Super, FP16
CTC inference returned blank text while FP32 returned text for the same sample;
set `OMNILINGUAL_DTYPE=float32` if you see that behavior. FP32 also needs more
VRAM, and this observation is not a transcription-accuracy claim.

To bake another upstream model into an image:

```bash
docker build \
  --target default \
  --build-arg MODEL_NAME=omniASR_LLM_300M_v2 \
  --tag omniasr-server:llm-300m .
```

Or use the guarded build helper:

```bash
bash build.sh
MODEL_NAME=omniASR_LLM_300M_v2 bash build.sh
```

`BACKEND`, `MODEL_NAME`, `NAMESPACE`, `LATEST_TAG`, and `PUSH` configure the
helper. `PUSH=false` (the default) loads the image locally; `PUSH=true` pushes
the resulting tag. Review a target registry before enabling a push.

## Optional CUDA image

The CUDA path is opt-in because the updated stack is not yet runtime-validated:

```bash
BACKEND=cuda bash build.sh
docker run --rm --gpus all \
  --publish 127.0.0.1:8080:8080 \
  omniasr-server:cu126-pt291-ctc-300m-v2
```

It locks PyTorch/torchaudio 2.9.1+cu126 to fairseq2/fairseq2n 0.8.1+cu126.
Do not describe this path as supported until the built image loads a model and
transcribes a fixed sample on the target GPU.

## Model and resource notes

- The standard CTC and LLM variants have an upstream 40-second input limit.
  Use an `Unlimited` LLM asset or split longer audio deliberately.
- The upstream table lists about 2 GiB of inference VRAM for CTC 300M v2 on an
  A100 using BF16. Different hardware, dtypes, drivers, and concurrent GPU
  workloads change that requirement.
- In one local CPU check, 262.596 seconds of audio split into ten bounded chunks
  completed in 60.079 seconds, peaked at about 2.493 GiB sampled RAM, and used
  roughly six logical CPU cores. This is a single observation, not a benchmark
  or quality guarantee.
- A successful language label, HTTP response, or non-empty transcript does not
  establish transcription accuracy. Review output with a fluent speaker.

## Security and deployment

This is a local inference wrapper, not a hardened public service. It has no
authentication, TLS termination, rate limiting, request-size limit, or tenant
isolation. Bind to `127.0.0.1` for local use. If you expose it to a network,
place it behind a trusted reverse proxy and add those controls. Uploaded files
are read into memory, so enforce a body-size limit at the proxy.

The container needs outbound network access while building or when asked to
load a model that was not baked into the image. A container using its preloaded
model can otherwise be run with a restricted network policy. Never mount the
Docker socket or a broad home directory into this service.

## License

The wrapper code is MIT licensed. Omnilingual ASR code and model weights are
published by Meta under Apache 2.0; review the upstream project for the exact
terms and model documentation.
