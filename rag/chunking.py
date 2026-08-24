def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
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
