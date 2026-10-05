# /// script
# dependencies = [
#     "openai==3.24.0",
# ]
# ///

"""Send one local audio file to an Omnilingual-ASR server."""

import argparse
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to a local audio file")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="omniASR_CTC_300M_v2",
        help="Model label sent to the endpoint (the server uses its loaded model)",
    )
    parser.add_argument(
        "--language",
        help="Optional language code; only LLM model variants use this hint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    request = {"model": args.model}
    if args.language:
        request["language"] = args.language

    with args.audio.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            **request,
        )

    print(transcription.text)


if __name__ == "__main__":
    main()
