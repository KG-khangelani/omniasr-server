# Dependency policy

This repository locks every Python dependency in `uv.lock` and pins every
direct dependency in `pyproject.toml`. CPU and CUDA are mutually exclusive
extras so a CPU install does not silently pull NVIDIA packages.

## Core compatibility matrix

| Component | Pin | Reason |
| --- | --- | --- |
| Python | 3.12 | Supported by Omnilingual ASR 0.2 and fairseq2 0.8 |
| Omnilingual ASR | 0.2.0 | Latest released package |
| PyTorch / torchaudio | 2.9.1 | Exact version supported by released fairseq2 0.8 wheels |
| fairseq2 / fairseq2n | 0.8.1 | Latest released fairseq2 version |
| CUDA variant | 12.6 | Official fairseq2/PyTorch 2.9.1 wheel variant |

`omnilingual-asr==0.2.0` still declares `fairseq2<=0.6.0`. This project uses an
explicit uv override to install fairseq2 0.8.1, whose released native wheels
match PyTorch 2.9.1. The CPU combination has passed model loading and a real
transcription check. The updated CUDA combination is locked but has not had a
runtime validation, so it remains opt-in.

Do not update PyTorch, torchaudio, fairseq2, or fairseq2n independently.
fairseq2 includes a native extension and requires the exact PyTorch ABI named
by its published wheel matrix. A resolver succeeding is not proof that a new
combination can load a model.

Official references:

- [fairseq2 installation matrix](https://github.com/facebookresearch/fairseq2#installing-on-linux)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Omnilingual ASR package metadata](https://pypi.org/project/omnilingual-asr/)

## Updating

1. Check the three official sources above for a released, matching matrix.
2. Update direct pins and both explicit uv indexes together.
3. Run `uv lock --upgrade`, then `uv lock --check`.
4. Run `uv sync --frozen --extra cpu` and `uv run --frozen --extra cpu pytest -q`.
5. Confirm the CPU environment contains no `nvidia-*` distributions.
6. Load an actual model and transcribe a fixed local sample before publishing.
7. Validate CUDA separately on compatible hardware before describing it as
   supported.

The GitHub workflow performs steps 3–5 for the CPU extra. Model weights are
not downloaded in CI.

## Known advisory constraint

As of 2026-10-05, `pip-audit` reports nine advisory records (six unique IDs:
[`PYSEC-2025-217`](https://osv.dev/vulnerability/PYSEC-2025-217),
[`PYSEC-2026-2288`](https://osv.dev/vulnerability/PYSEC-2026-2288),
[`PYSEC-2026-2289`](https://osv.dev/vulnerability/PYSEC-2026-2289),
[`PYSEC-2026-2290`](https://osv.dev/vulnerability/PYSEC-2026-2290),
[`PYSEC-2026-3929`](https://osv.dev/vulnerability/PYSEC-2026-3929), and
[`PYSEC-2026-4174`](https://osv.dev/vulnerability/PYSEC-2026-4174)) against
`transformers==4.57.6`. That is the newest version allowed by fairseq2 0.8.1's
declared `transformers~=4.57` requirement; available patched releases are in
the incompatible 5.x series. Overriding another model-stack constraint without
upstream support would not be a safe update.

The reported paths involve loading or saving attacker-controlled model,
configuration, tokenizer, or custom-code content. This server's HTTP API
accepts audio bytes only: it does not accept model repositories, checkpoints,
or configuration objects. Keep model selection under operator control, use
trusted upstream asset cards, bake the model into the image, and run the
preloaded container without outbound network access where possible. Do not
adapt this wrapper to load untrusted model content while this constraint
remains. Re-check the advisories when fairseq2 publishes a Transformers 5
compatible release.
