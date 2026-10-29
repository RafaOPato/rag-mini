import pytest

from rag.chunking import chunk_corpus, chunk_text, load_sections, overlap_boundaries
from rag.pipeline import ROOT

TEXT = " ".join(f"Esta é a frase número {i} do texto de teste." for i in range(1, 21))


def test_respeita_tamanho_maximo():
    assert all(len(c) <= 200 for c in chunk_text(TEXT, size=200, overlap=50))


def test_trechos_vizinhos_se_sobrepoem():
    chunks = chunk_text(TEXT, size=200, overlap=50)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = a.split(". ")[-1].rstrip(".")
        assert last_sentence in b  # a última frase de um trecho abre o próximo


def test_nenhuma_frase_se_perde():
    joined = " ".join(chunk_text(TEXT, size=200, overlap=50))
    assert all(f"frase número {i} " in joined for i in range(1, 21))


def test_frase_gigante_e_cortada():
    assert all(len(c) <= 50 for c in chunk_text("palavra " * 100, size=50, overlap=10))


def test_overlap_invalido():
    with pytest.raises(ValueError):
        chunk_text("a. b.", size=10, overlap=10)


def test_corpus_tem_metadados():
    sections = load_sections(ROOT / "corpus")
    assert 12 <= len(sections) <= 15
    chunks = chunk_corpus(ROOT / "corpus")
    assert {c.source for c in chunks} == {s[0] for s in sections}
    assert all(c.section and c.text for c in chunks)
    assert chunks[0].embed_text.startswith(chunks[0].section)


def test_sobreposicao_padrao_repete_texto_na_maioria_das_fronteiras():
    repeated, total = overlap_boundaries(ROOT / "corpus")
    assert total > 20 and repeated / total >= 0.8
    assert overlap_boundaries(ROOT / "corpus", overlap=0)[0] == 0  # sem overlap, nada se repete
