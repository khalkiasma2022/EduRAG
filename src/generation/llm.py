"""Génération locale avec un modèle Hugging Face.

Le modèle est téléchargé puis exécuté dans Python, par exemple sur Google Colab.
Aucune API distante n'est appelée, et la réponse n'est jamais exécutée comme du code.
"""

import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Modèle instructif léger, prévu pour tourner sur un GPU de Colab.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1024
# Plafond pour éviter une génération trop longue et trop coûteuse en mémoire.
MAX_NEW_TOKENS_LIMIT = 2048


class LLM:
    """Modèle de langage chargé une seule fois."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Choisit le modèle et le processeur, puis charge les poids."""
        load_dotenv()
        self.model_name = _model_name(model_name)
        self.device = _select_device(device)
        self.tokenizer = None
        self.model = None
        self.load()

    def load(self) -> None:
        """Charge le tokenizer et le modèle.

        float16 sur GPU réduit la mémoire. float32 sur CPU reste compatible.
        """
        self.tokenizer = None
        self.model = None
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                weights_only=True,
            )
            model = model.to(self.device)
            model.eval()
        except Exception as error:
            self.tokenizer = None
            self.model = None
            raise RuntimeError(f"Impossible de charger le modèle {self.model_name}.") from error

        # Certains modèles n'ont pas de jeton de padding : on réutilise la fin de séquence.
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.model = model

    def is_loaded(self) -> bool:
        """Indique si le tokenizer et le modèle sont prêts."""
        return self.tokenizer is not None and self.model is not None

    def generate(self, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Renvoie uniquement le texte nouveau produit à partir du prompt."""
        if not self.is_loaded():
            raise RuntimeError("Le modèle n'est pas chargé.")

        _check_prompt(prompt)
        _check_max_new_tokens(max_new_tokens)
        encoded = self._encode_prompt(prompt)
        encoded = _move_to_device(encoded, self.device)
        input_length = encoded["input_ids"].shape[-1]

        try:
            # Pas de gradients : la génération consomme moins de mémoire.
            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            new_token_ids = output_ids[0, input_length:]
            # On décode seulement les jetons nouveaux, pas le prompt.
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        except Exception as error:
            raise RuntimeError("La génération a échoué.") from error

        return text.strip()

    def _encode_prompt(self, prompt: str):
        """Prépare le prompt avec le format de conversation du modèle si possible."""
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as error:
                raise RuntimeError("La génération a échoué.") from error

        try:
            return self.tokenizer(prompt, return_tensors="pt")
        except Exception as error:
            raise RuntimeError("La génération a échoué.") from error


def _model_name(model_name: str | None) -> str:
    """Lit le nom demandé, sinon la variable d'environnement, sinon le modèle par défaut."""
    if model_name is None:
        model_name = os.environ.get("EDURAG_LLM_MODEL", DEFAULT_MODEL_NAME)
    if not isinstance(model_name, str) or model_name.strip() == "":
        raise ValueError("Le nom du modèle ne peut pas être vide.")
    return model_name.strip()


def _select_device(device: str | None) -> str:
    """Utilise le GPU CUDA lorsqu'il est disponible, sinon le CPU."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("Le processeur demandé doit être 'cpu' ou 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA est demandé, mais aucun GPU n'est disponible.")
    return device


def _check_prompt(prompt: object) -> None:
    if not isinstance(prompt, str):
        raise TypeError("Le prompt doit être une chaîne de caractères.")
    if prompt.strip() == "":
        raise ValueError("Le prompt ne peut pas être vide.")


def _check_max_new_tokens(max_new_tokens: object) -> None:
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens doit être un entier positif.")
    if max_new_tokens <= 0 or max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        raise ValueError(
            f"max_new_tokens doit être compris entre 1 et {MAX_NEW_TOKENS_LIMIT}."
        )


def _move_to_device(encoded, device: str):
    """Place les tenseurs du prompt sur le CPU ou le GPU."""
    if hasattr(encoded, "to") and not isinstance(encoded, dict):
        return encoded.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
