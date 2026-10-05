#!/usr/bin/env bash

set -euo pipefail

# Build a Linux/amd64 image with its selected model preloaded. The CPU image is
# the supported default; BACKEND=cuda opts into the CUDA 12.6 target.
MODEL_NAME="${MODEL_NAME:-omniASR_CTC_300M_v2}"
BACKEND="${BACKEND:-cpu}"
NAMESPACE="${NAMESPACE:-}"
LATEST_TAG="${LATEST_TAG:-false}"
PUSH="${PUSH:-false}"

if [[ ! "$MODEL_NAME" =~ ^omniASR_[A-Za-z0-9_]+$ ]]; then
    echo "Invalid MODEL_NAME=$MODEL_NAME" >&2
    exit 2
fi

case "$BACKEND" in
    cpu)
        target="default"
        base_tag="cpu-pt291"
        run_flags=()
        ;;
    cuda)
        target="cuda"
        base_tag="cu126-pt291"
        run_flags=(--gpus all)
        ;;
    *)
        echo "Unsupported BACKEND=$BACKEND (expected cpu or cuda)" >&2
        exit 2
        ;;
esac

tag_suffix="$({ printf '%s' "$MODEL_NAME" | sed 's/^omniASR_//' | tr 'A-Z_' 'a-z-'; })"
if [[ -n "$NAMESPACE" ]]; then
    image_name="${NAMESPACE%/}/omniasr-server"
else
    image_name="omniasr-server"
fi
image_tag="$image_name:$base_tag-$tag_suffix"

build_args=(
    docker buildx build
    --platform linux/amd64
    --target "$target"
    --build-arg "MODEL_NAME=$MODEL_NAME"
    -t "$image_tag"
)

if [[ "$LATEST_TAG" == "true" ]]; then
    build_args+=(-t "$image_name:latest")
elif [[ "$LATEST_TAG" != "false" ]]; then
    echo "LATEST_TAG must be true or false" >&2
    exit 2
fi

if [[ "$PUSH" == "true" ]]; then
    build_args+=(--push)
elif [[ "$PUSH" == "false" ]]; then
    build_args+=(--load)
else
    echo "PUSH must be true or false" >&2
    exit 2
fi

"${build_args[@]}" .

printf 'Built %s\nRun it with:\n  docker run' "$image_tag"
if ((${#run_flags[@]})); then
    printf ' %q' "${run_flags[@]}"
fi
printf ' --publish 127.0.0.1:8080:8080 %q\n' "$image_tag"
