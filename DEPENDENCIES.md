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
| Transformers | 5.18.0 | Latest compatible version; clears the 4.57.6 advisories |
| Hugging Face Hub | 1.33.0 | Version selected by the Transformers 5 lock |
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

## fairseq2 Transformers 5 metadata backport

Released fairseq2 0.8.1 declares `transformers~=4.57`, which prevents a
resolver from selecting releases that fix the known 4.57.6 advisories.
fairseq2 upstream subsequently merged
[PR #1508](https://github.com/facebookresearch/fairseq2/pull/1508) in
[commit `027bdeb`](https://github.com/facebookresearch/fairseq2/commit/027bdebca4b9177f1ac1df1cc981b9288465d68e),
updating its metadata to Transformers 5 and loosening the Hugging Face Hub
upper bound. The change passed upstream's Python 3.12 / PyTorch 2.9.1 CPU,
CUDA 12.6, and CUDA 12.8 jobs.

Until fairseq2 publishes a release containing that change, this project uses
a uv package-scoped override for fairseq2 0.8.1 only. It permits
`transformers>=5.10,<6` and `huggingface-hub>=0.32,<2`; the 5.10 floor is the
first release above all known affected Transformers versions as of
2026-10-05. The override changes dependency metadata only. It does not patch
or vendor fairseq2 code, and it does not remove Transformers: fairseq2 imports
its Hugging Face integration while initializing the model composition layer.

`uv audit --locked` is part of CI. Model sources still remain operator
controlled; this server accepts audio bytes, not model repositories,
checkpoints, configuration objects, or custom code.

## Reviewed PyTorch audit exceptions

The locked PyTorch 2.9.1 build currently has four audit findings:

- [`PYSEC-2026-139`](https://osv.dev/vulnerability/PYSEC-2026-139), local
  deserialization through the PT2 loading handler; the database provides no
  fixed-version marker.
- [`GHSA-qfhq-4f3w-5fph`](https://osv.dev/vulnerability/GHSA-qfhq-4f3w-5fph),
  local memory corruption through `torch.lstm_cell`, fixed in PyTorch 2.10.
- [`GHSA-rrmf-rvhw-rf47`](https://osv.dev/vulnerability/GHSA-rrmf-rvhw-rf47),
  local memory corruption through `torch.jit.script`, fixed in PyTorch 2.13.
- [`PYSEC-2026-2286`](https://osv.dev/vulnerability/PYSEC-2026-2286), crafted
  checkpoint memory corruption in `torch.load(..., weights_only=True)`, fixed
  in PyTorch 2.10.

These are explicit CI exceptions, not claims that PyTorch 2.9.1 is patched.
The released fairseq2 0.8 matrix has no build newer than PyTorch 2.9.1, and
fairseq2 warns that its native extension must exactly match PyTorch's ABI.
The wrapper does not accept checkpoints or Python/model code through HTTP;
builds preload an operator-selected upstream asset and production runs should
have no outbound network. CI continues to fail on every advisory except these
four IDs, and `PYSEC-2026-139` uses `--ignore-until-fixed` so a published fix
automatically makes that exception fail. Revisit all four when fairseq2 ships
an ABI-compatible release on a patched PyTorch line.
