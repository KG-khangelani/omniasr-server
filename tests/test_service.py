"""Tests for language mapping functionality."""

from unittest.mock import patch

import pytest
import torch

from app.service import OmnilingualASRService


class TestOmnilingualASRService:
    """Tests for the OmnilingualASRService class."""

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("omniASR_W2V_300M", False),
            ("omniASR_W2V_1B", False),
            ("omniASR_W2V_3B", False),
            ("omniASR_W2V_7B", False),
            ("omniASR_CTC_300M", False),
            ("omniASR_CTC_1B", False),
            ("omniASR_CTC_3B", False),
            ("omniASR_CTC_7B", False),
            ("omniASR_CTC_300M_v2", False),
            ("omniASR_CTC_1B_v2", False),
            ("omniASR_CTC_3B_v2", False),
            ("omniASR_CTC_7B_v2", False),
            ("omniASR_LLM_300M", True),
            ("omniASR_LLM_1B", True),
            ("omniASR_LLM_3B", True),
            ("omniASR_LLM_7B", True),
            ("omniASR_LLM_300M_v2", True),
            ("omniASR_LLM_1B_v2", True),
            ("omniASR_LLM_3B_v2", True),
            ("omniASR_LLM_7B_v2", True),
            ("omniASR_LLM_Unlimited_300M_v2", True),
            ("omniASR_LLM_Unlimited_1B_v2", True),
            ("omniASR_LLM_Unlimited_3B_v2", True),
            ("omniASR_LLM_Unlimited_7B_v2", True),
            ("omniASR_LLM_7B_ZS", True),
        ],
    )
    def test_is_llm_model(self, model_name: str, expected: bool):
        """Test that the is_llm_model property returns the correct value."""
        with patch("app.service.MODEL_NAME", model_name):
            service = OmnilingualASRService()
            assert service.is_llm_model == expected

    @patch("app.service.ASRInferencePipeline")
    @patch("app.service.torch.backends.mps.is_available", return_value=False)
    @patch("app.service.torch.cuda.is_available", return_value=False)
    def test_cpu_defaults_to_float32(self, _cuda, _mps, pipeline, monkeypatch):
        monkeypatch.delenv("OMNILINGUAL_DTYPE", raising=False)
        service = OmnilingualASRService()
        service.load_model()

        assert pipeline.call_args.kwargs["dtype"] is torch.float32

    @patch.dict("os.environ", {"OMNILINGUAL_DTYPE": "float32"})
    @patch("app.service.ASRInferencePipeline")
    @patch("app.service.torch.cuda.get_device_capability", return_value=(7, 5))
    @patch("app.service.torch.cuda.is_available", return_value=True)
    def test_dtype_override(self, _available, _capability, pipeline):
        service = OmnilingualASRService()
        service.load_model()

        assert pipeline.call_args.kwargs["dtype"] is torch.float32

    @patch.dict("os.environ", {"OMNILINGUAL_DTYPE": "invalid"})
    @patch("app.service.torch.backends.mps.is_available", return_value=False)
    @patch("app.service.torch.cuda.is_available", return_value=False)
    def test_invalid_dtype_override_is_rejected(self, _cuda, _mps):
        service = OmnilingualASRService()

        with pytest.raises(ValueError, match="Invalid OMNILINGUAL_DTYPE"):
            service.load_model()
