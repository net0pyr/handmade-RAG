from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Единица поиска: текст плюс достаточно метаданных, чтобы сослаться на источник."""

    text: str
    source: str      # имя документа, из которого пришёл чанк
    position: int    # порядковый номер чанка внутри этого документа

    @property
    def label(self) -> str:
        return f"{self.source}, фрагмент {self.position + 1}"


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into ~size-character pieces with an overlap.
    Nudge the boundary to the nearest separator so we don't cut mid-sentence."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end]
        if end < n:                              # не обрезаем последнюю часть
            for sep in ("\n\n", ". ", ".\n", "\n", " "):
                pos = piece.rfind(sep)
                if pos > size * 0.5:             # граница найдена не слишком рано
                    piece = piece[: pos + len(sep)]
                    end = start + len(piece)
                    break
        cleaned = piece.strip()
        if cleaned:
            chunks.append(cleaned)
        if end >= n:
            break
        start = max(end - overlap, start + 1)    # +1 защита от бесконечного цикла
    return chunks


def chunk_document(source: str, text: str, **kwargs) -> list[Chunk]:
    """То же разбиение, но с сохранением имени документа и позиции чанка в нём."""
    return [
        Chunk(text=piece, source=source, position=i)
        for i, piece in enumerate(chunk_text(text, **kwargs))
    ]
