import math

import pytest

from rag.chunking import Chunk
from rag.embedders import HashEmbedder, stem, tokens
from rag.pipeline import ingest
from rag.stores import MemoryStore, cosine


def test_hash_e_deterministico_e_normalizado():
    e = HashEmbedder()
    a, b = e.embed(["Escoramento de valas"])[0], e.embed(["Escoramento de valas"])[0]
    assert a == b
    assert len(a) == e.dim
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_hash_ignora_acento_caixa_e_plural():
    e = HashEmbedder()
    a, b = e.embed(["Fôrmas LIMPAS"])[0], e.embed(["formas limpa"])[0]
    assert cosine(a, b) > 0.99


def test_hash_texto_parecido_pontua_mais():
    e = HashEmbedder()
    q, perto, longe = e.embed(["altura do guarda-corpo", "o guarda-corpo tem 1,20 m de altura", "cimento em pilhas"])
    assert cosine(q, perto) > cosine(q, longe)


def test_tokens_e_stem():
    assert tokens("O Concreto é lançado") == ["concreto", "lancado"]
    assert stem("escoramento") == stem("escoramentos") == "escora"


def test_memory_store_ordena_por_cosseno_e_respeita_k():
    store = MemoryStore()
    chunks = [Chunk("a.md", "A", 0, "x"), Chunk("b.md", "B", 0, "y"), Chunk("c.md", "C", 0, "z")]
    store.reset(2)
    store.add(chunks, [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    hits = store.search([1.0, 0.1], k=2)
    assert [h.source for h in hits] == ["a.md", "b.md"]
    assert hits[0].score > hits[1].score
    assert hits[0].section == "A" and hits[0].text == "x"


def test_cosine_normaliza_pelas_normas():
    assert cosine([2.0, 0.0], [3.0, 0.0]) == pytest.approx(1.0)


def test_cosine_vetor_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_ingest_com_corpus_vazio_ou_inexistente_nao_mexe_no_indice(tmp_path):
    class StoreEspiao:
        chamadas = []

        def reset(self, dim):
            self.chamadas.append("reset")

        def add(self, chunks, vectors):
            self.chamadas.append("add")

    store = StoreEspiao()
    for pasta in (tmp_path, tmp_path / "pasta-com-erro-de-digitacao"):
        with pytest.raises(SystemExit):
            ingest(store, HashEmbedder(), pasta)
    assert store.chamadas == []
