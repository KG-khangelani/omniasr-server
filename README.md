# Omnilingual-ASR Model Server

A FastAPI-based ASR model server for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) with an OpenAI Whisper-compatible API.

## Quick Start

### Local Development

```bash
# Install dependencies with uv
uv sync --extra cpu

# Run the server
uv run --extra cpu python main.py
```

## Building a Docker Image

The default image is CPU-only. It uses PyTorch 2.9.1 and fairseq2 0.8.1 without CUDA or NVIDIA wheel dependencies. An opt-in CUDA 12.6 image uses the matching PyTorch 2.9.1/fairseq2 0.8.1 wheel matrix.

Omnilingual ASR 0.2.0 still declares `fairseq2<=0.6.0`; this project carries an explicit override to fairseq2 0.8.1. Treat both backends as compatibility candidates until model-load and transcription checks pass for the built image.

### Helpful resources

- Supported combinations of CUDA, PyTorch, and Python for `fairseq2`: https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#variants
- Organizing sources and indices: https://docs.astral.sh/uv/concepts/indexes/

### Using the build script

The [`build.sh`](build.sh) script is a good place to start your own builds:

```bash
# Build with default model
bash build.sh

# Build the optional CUDA image
BACKEND=cuda bash build.sh

# Build with variant model, tag as latest, and push
MODEL_NAME=omniASR_LLM_1B_v2 LATEST_TAG=true PUSH=true bash build.sh

# Build with namespace
NAMESPACE=abc bash build.sh

# Build with namespace and push
NAMESPACE=abc PUSH=true bash build.sh
```

**Build script options:**

- `MODEL_NAME` - Name of the model to build (default: `omniASR_LLM_300M_v2`)
- `BACKEND` - `cpu` (default) or `cuda`
- `NAMESPACE` - Namespace/registry prefix for the image name (optional). If provided, images will be tagged as `NAMESPACE/omniasr-server`. If not provided, defaults to `omniasr-server`
- `LATEST_TAG` - Set to `"true"` to also tag the image as `latest` (default: `false`)
- `PUSH` - Set to `"true"` to push the image to the registry after building (default: `false`)

The image tag begins with `cpu-pt291` by default or `cu126-pt291` for `BACKEND=cuda`, followed by the model suffix.

### Manual build

You can also build manually using Docker:

```bash
docker build --build-arg MODEL_NAME=omniASR_LLM_300M_v2 -t omniasr-server .
```

Then, run the CPU image:

```bash
docker run -p 8080:8080 omniasr-server
```

Build and run the optional CUDA image explicitly:

```bash
docker build --target cuda --build-arg MODEL_NAME=omniASR_LLM_300M_v2 -t omniasr-server:cuda .
docker run --gpus all -p 8080:8080 omniasr-server:cuda
```

## API Usage

The API is (somewhat) compatible with OpenAI's Whisper transcription endpoint. Some parameters (like `model`) are ignored (for now!) since the server only hosts one model and Omnilingual-ASR doesn't have all the features of Whisper.

### Transcribe Audio

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=omniASR_CTC_300M_v2"
```

### With Language Hint

This works for the LLM variants only. For CTC and W2V, the `language` parameter is ignored.

**ISO 639-1 (OpenAI API native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=en"
```

**ISO 639-3 / Script (Omnilingual-ASR native)**

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=omniASR_LLM_1B_v2" \
  -F "language=eng_Latn"
```

Languages are mapped heuristically from ISO 639-1 (Whisper's API) to Omnilingual-ASR's format. See how it's mapped in [`app/languages.py`](app/languages.py). For the best results, use Omnilingual-ASR's language codes.

### Response Formats

**JSON (default)**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions -F "file=@audio.wav"
```
```json
{"text": "Hello, world!"}
```

**Plain Text**
```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

### Python Client

```bash
uv run scripts/openai_client.py
```

See the [openai_client.py](scripts/openai_client.py) code. It's pretty straightforward.


## Configuration


### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `omniASR_CTC_300M_v2` | Model to use for transcription |
| `OMNILINGUAL_PORT` | `8080` | Server port |
| `OMNILINGUAL_HOST` | `0.0.0.0` | Server host |
| `OMNILINGUAL_DTYPE` | `auto` | Model dtype: `auto`, `float16`, `bfloat16`, or `float32`. Use `float32` if FP16 CTC inference produces blank output on older GPUs. CPU defaults to FP32. |

### Changing the Model

See [Omnilingual-ASR's GitHub page](https://github.com/facebookresearch/omnilingual-asr/tree/main?tab=readme-ov-file#model-architectures) for a list of available models.

You can specify the model either at build time or at runtime:

**At build time (recommended):**

```bash
# Build with a specific model
MODEL_NAME=omniASR_LLM_1B_v2 bash build.sh

# Then run the container
docker run -p 8080:8080 omniasr-server:cpu-pt291-llm-1b-v2
```

**At runtime:**

```bash
# Run with a different model (model will be downloaded on first run)
docker run -p 8080:8080 \
  -e MODEL_NAME=omniASR_CTC_1B_v2 \
  omniasr-server
```

**When running locally:**

```bash
MODEL_NAME=omniASR_CTC_1B_v2 uv run --extra cpu python main.py
```

**NOTE:** When running locally, on the first run, `fairseq` will download the weights and cache it to your device. Subsequent runs only loads the cached weights.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio file |
| `/v1/models` | GET | List the deployed model |
| `/health-check` | GET | Health check |

## License

This server code is MIT licensed. The Omnilingual ASR models are released under Apache 2.0 by Meta.

