"""Embeddings: modelo local (fastembed) ou um hashing determinístico para testes.

Os dois devolvem vetores normalizados (norma 1). `min_score` é o limiar de cosseno
abaixo do qual o `ask.py` se recusa a responder (calibrado em eval/, ver README).
"""
import math
import re
import sys
import unicodedata
import zlib
from collections import Counter

# Palavras que não ajudam a achar nada (já sem acento, em minúsculas).
STOPWORDS = set(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem que e ou se ao aos "
    "ser sao foi ha mais menos como qual quais quanto quantos quantas quem onde quando deve devem "
    "pode podem pelo pela pelos pelas seu sua seus suas este esta esse essa isso entre ate ja nao "
    "so tambem muito e eh vai sobre depois antes".split()
)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def tokens(text: str) -> list[str]:
    """minúsculas, sem acento, só letras/números, sem stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS]


def stem(token: str) -> str:
    # Stemmer grosseiro: tira o "s" do plural e fica com as 6 primeiras letras.
    # "escoramento" e "escorar" viram "escora". Erra bastante, mas é estável.
    if len(token) > 3:
        token = token.rstrip("s")
    return token[:6]


class HashEmbedder:
    """Bag-of-words por 'hashing trick' (512 dimensões, palavras + pares de palavras).

    Determinístico e sem dependências: serve para os testes e para rodar sem baixar
    modelo. É lexical: só acha o que compartilha palavras (depois do stemmer) com a pergunta.
    """

    name = "hash"
    dim = 512
    min_score = 0.05  # quase desligado de propósito: no eval nenhum limiar separa as perguntas (ver README)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        words = [stem(t) for t in tokens(text)]
        feats = Counter(words + [f"{a}_{b}" for a, b in zip(words, words[1:])])
        vec = [0.0] * self.dim
        for feat, count in feats.items():
            h = zlib.crc32(feat.encode())
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(count))
        return _normalize(vec)


class FastEmbedder:
    """Modelo multilíngue local via fastembed (ONNX, sem GPU, sem API)."""

    name = "fastembed"
    dim = 384
    min_score = 0.25  # qualquer valor entre 0.13 e 0.29 dá o mesmo resultado no eval (ver README)
    MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        from fastembed import TextEmbedding  # import tardio: é opcional

        self._model = TextEmbedding(model_name=self.MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_normalize([float(x) for x in v]) for v in self._model.embed(texts)]


def get_embedder(name: str = "auto"):
    if name == "hash":
        return HashEmbedder()
    if name == "fastembed":
        return FastEmbedder()
    if name != "auto":
        raise ValueError(f"embedder desconhecido: {name}")
    try:
        import fastembed  # noqa: F401
    except ImportError:
        print("aviso: fastembed não instalado, usando o embedder 'hash' (lexical).", file=sys.stderr)
        return HashEmbedder()
    return FastEmbedder()
