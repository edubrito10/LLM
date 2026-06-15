def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extrai texto de um ficheiro PDF ou Word.

    Args:
        file_bytes: conteúdo do ficheiro em bytes
        filename:   nome do ficheiro (usado para detetar o tipo)

    Returns:
        Texto extraído como string.

    Raises:
        ValueError: se o formato não for suportado
    """
    if filename.lower().endswith(".pdf"):
        return _extract_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        return _extract_docx(file_bytes)
    else:
        raise ValueError(f"Formato não suportado: {filename}")


def _extract_pdf(file_bytes: bytes) -> str:
    """Usa PyMuPDF para extrair texto de um PDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    """Usa python-docx para extrair texto de um ficheiro Word."""
    import io
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """
    Divide o texto em chunks de tamanho aproximado (em palavras),
    com overlap entre chunks consecutivos para preservar contexto.

    Args:
        text:       texto completo do documento
        chunk_size: número máximo de palavras por chunk
        overlap:    número de palavras de sobreposição entre chunks

    Returns:
        Lista de strings (chunks).
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
