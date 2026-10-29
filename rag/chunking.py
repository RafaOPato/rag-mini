"""Leitura do corpus e divisão em trechos (chunks) com sobreposição."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    source: str   # nome do arquivo, ex.: 08-concretagem-e-cura.md
    section: str  # título da seção (primeira linha "# ...")
    index: int    # posição do trecho dentro da seção
    text: str

    @property
    def embed_text(self) -> str:
        # O título entra no texto que vira vetor: ajuda trechos que não repetem o assunto.
        return f"{self.section}\n{self.text}"


def load_sections(corpus_dir) -> list[tuple[str, str, str]]:
    """Devolve (arquivo, título, corpo) para cada .md do corpus, em ordem alfabética."""
    sections = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path.name}: a primeira linha deve ser '# Título'")
        sections.append((path.name, lines[0][2:].strip(), "\n".join(lines[1:]).strip()))
    return sections


def _units(text: str, size: int) -> list[str]:
    """Quebra o texto em frases/linhas; nenhuma unidade passa de `size` caracteres."""
    units = []
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            while len(sentence) > size:  # frase gigante: corta nas palavras
                cut = sentence.rfind(" ", 0, size)
                cut = cut if cut > 0 else size
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)
    return units


def chunk_text(text: str, size: int = 400, overlap: int = 150) -> list[str]:
    """Junta frases até ~`size` caracteres; cada trecho repete a(s) última(s) frase(s)
    do anterior (até `overlap` caracteres) para não cortar uma ideia ao meio."""
    if overlap >= size:
        raise ValueError("overlap precisa ser menor que size")
    chunks, cur = [], []
    for unit in _units(text, size):
        if cur and len(" ".join(cur + [unit])) > size:
            chunks.append(" ".join(cur))
            tail = []
            for u in reversed(cur):
                if len(" ".join([u] + tail)) > overlap:
                    break
                tail.insert(0, u)
            cur = tail if len(" ".join(tail + [unit])) <= size else []
        cur.append(unit)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_corpus(corpus_dir, size: int = 400, overlap: int = 150) -> list[Chunk]:
    return [
        Chunk(source, title, i, text)
        for source, title, body in load_sections(corpus_dir)
        for i, text in enumerate(chunk_text(body, size, overlap))
    ]


def overlap_boundaries(corpus_dir, size: int = 400, overlap: int = 150) -> tuple[int, int]:
    """(fronteiras entre trechos vizinhos que repetem texto, total de fronteiras)."""
    repeated = total = 0
    for _, _, body in load_sections(corpus_dir):
        chunks = chunk_text(body, size, overlap)
        for a, b in zip(chunks, chunks[1:]):
            total += 1
            repeated += _units(b, size)[0] in a  # o trecho seguinte abre com texto do anterior?
    return repeated, total
