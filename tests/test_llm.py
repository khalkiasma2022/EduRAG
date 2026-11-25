"""Tests du modèle de langage local.

AutoTokenizer et AutoModelForCausalLM sont simulés.
Le vrai modèle Qwen n'est pas téléchargé.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.generation.llm import DEFAULT_MODEL_NAME, LLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _fake_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.chat_template = "template"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    input_ids = torch.tensor([[1, 2, 3, 4]])
    tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    tokenizer.decode.return_value = "Réponse simulée"
    return tokenizer


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 10, 11]])
    return model


@pytest.fixture
def llm_parts():
    tokenizer = _fake_tokenizer()
    model = _fake_model()
    with (
        patch("src.generation.llm.AutoTokenizer") as tokenizer_cls,
        patch("src.generation.llm.AutoModelForCausalLM") as model_cls,
        patch("src.generation.llm.torch.cuda.is_available", return_value=False),
    ):
        tokenizer_cls.from_pretrained.return_value = tokenizer
        model_cls.from_pretrained.return_value = model
        language_model = LLM(model_name=MODEL_NAME, device="cpu")
    return language_model, tokenizer_cls, model_cls, tokenizer, model


def test_load_model_with_mocks(llm_parts) -> None:
    language_model, tokenizer_cls, model_cls, tokenizer, model = llm_parts

    tokenizer_cls.from_pretrained.assert_called_once_with(MODEL_NAME)
    assert "token" not in tokenizer_cls.from_pretrained.call_args.kwargs
    model_cls.from_pretrained.assert_called_once_with(
        MODEL_NAME,
        dtype=torch.float32,
        weights_only=True,
    )
    model.to.assert_called_once_with("cpu")
    assert language_model.is_loaded()
    assert language_model.model is model
    assert language_model.tokenizer is tokenizer
    assert language_model.model_name == MODEL_NAME


def test_generate_returns_only_new_text(llm_parts) -> None:
    language_model, _tokenizer_cls, _model_cls, tokenizer, model = llm_parts

    answer = language_model.generate("Explique le cours")

    assert answer == "Réponse simulée"
    decoded_ids = tokenizer.decode.call_args.args[0]
    assert decoded_ids.tolist() == [10, 11]
    assert model.generate.call_args.kwargs["max_new_tokens"] == 1024
    tokenizer.apply_chat_template.assert_called_once()
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"] is True


def test_model_is_not_reloaded(llm_parts) -> None:
    language_model, tokenizer_cls, model_cls, _tokenizer, _model = llm_parts

    language_model.generate("Première question")
    language_model.generate("Deuxième question")

    assert tokenizer_cls.from_pretrained.call_count == 1
    assert model_cls.from_pretrained.call_count == 1


def test_cpu_is_selected_without_gpu() -> None:
    with (
        patch("src.generation.llm.AutoTokenizer") as tokenizer_cls,
        patch("src.generation.llm.AutoModelForCausalLM") as model_cls,
        patch("src.generation.llm.torch.cuda.is_available", return_value=False),
    ):
        tokenizer_cls.from_pretrained.return_value = _fake_tokenizer()
        model = _fake_model()
        model_cls.from_pretrained.return_value = model
        language_model = LLM(model_name=MODEL_NAME)

    assert language_model.device == "cpu"
    assert model_cls.from_pretrained.call_args.kwargs["dtype"] == torch.float32


def test_gpu_is_selected_when_available() -> None:
    with (
        patch("src.generation.llm.AutoTokenizer") as tokenizer_cls,
        patch("src.generation.llm.AutoModelForCausalLM") as model_cls,
        patch("src.generation.llm.torch.cuda.is_available", return_value=True),
    ):
        tokenizer_cls.from_pretrained.return_value = _fake_tokenizer()
        model = _fake_model()
        model_cls.from_pretrained.return_value = model
        language_model = LLM(model_name=MODEL_NAME)

    assert language_model.device == "cuda"
    model.to.assert_called_once_with("cuda")
    assert model_cls.from_pretrained.call_args.kwargs["dtype"] == torch.float16


@pytest.mark.parametrize(
    "prompt",
    ["", "   ", None, 12],
)
def test_invalid_prompts_raise(llm_parts, prompt) -> None:
    language_model, _tokenizer_cls, _model_cls, _tokenizer, _model = llm_parts

    with pytest.raises((TypeError, ValueError)):
        language_model.generate(prompt)


def test_max_new_tokens_limit_accepts_2048(llm_parts) -> None:
    language_model, _tokenizer_cls, _model_cls, _tokenizer, model = llm_parts

    language_model.generate("Question", max_new_tokens=2048)

    assert model.generate.call_args.kwargs["max_new_tokens"] == 2048


@pytest.mark.parametrize("max_new_tokens", [0, -1, True, False, 1.5, "512", 2049])
def test_invalid_max_new_tokens_raise(llm_parts, max_new_tokens) -> None:
    language_model, _tokenizer_cls, _model_cls, _tokenizer, _model = llm_parts

    with pytest.raises(ValueError):
        language_model.generate("Question", max_new_tokens=max_new_tokens)


def test_generation_error_is_reported(llm_parts) -> None:
    language_model, _tokenizer_cls, _model_cls, _tokenizer, model = llm_parts
    model.generate.side_effect = RuntimeError("mémoire insuffisante")

    with pytest.raises(RuntimeError, match="génération"):
        language_model.generate("Question")


def test_load_error_is_reported() -> None:
    with (
        patch("src.generation.llm.AutoTokenizer") as tokenizer_cls,
        patch("src.generation.llm.AutoModelForCausalLM") as model_cls,
        patch("src.generation.llm.torch.cuda.is_available", return_value=False),
    ):
        tokenizer_cls.from_pretrained.side_effect = OSError("modèle introuvable")
        model_cls.from_pretrained.return_value = _fake_model()

        with pytest.raises(RuntimeError, match="charger"):
            LLM(model_name=MODEL_NAME, device="cpu")


def test_model_name_can_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDURAG_LLM_MODEL", "org/modele-de-test")
    with (
        patch("src.generation.llm.AutoTokenizer") as tokenizer_cls,
        patch("src.generation.llm.AutoModelForCausalLM") as model_cls,
        patch("src.generation.llm.torch.cuda.is_available", return_value=False),
    ):
        tokenizer_cls.from_pretrained.return_value = _fake_tokenizer()
        model_cls.from_pretrained.return_value = _fake_model()
        language_model = LLM(device="cpu")

    assert language_model.model_name == "org/modele-de-test"
    tokenizer_cls.from_pretrained.assert_called_once_with("org/modele-de-test")


def test_default_model_name() -> None:
    assert DEFAULT_MODEL_NAME == MODEL_NAME


@pytest.mark.skipif(
    os.environ.get("EDURAG_RUN_REAL_LLM") != "1",
    reason=(
        "Test réel facultatif. Le lancer avec EDURAG_RUN_REAL_LLM=1 "
        "sur une machine qui peut télécharger Qwen."
    ),
)
def test_real_model_when_available() -> None:
    language_model = LLM(device="cpu")
    answer = language_model.generate("Dis bonjour en une courte phrase.", max_new_tokens=16)

    assert language_model.is_loaded()
    assert isinstance(answer, str)
    assert answer.strip() != ""
