def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return _extract_pdf(file_bytes)
    elif filename.lower().endswith(".docx"):
        return _extract_docx(file_bytes)
    else:
        raise ValueError(f"Formato não suportado: {filename}")


def _extract_pdf(file_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    import io
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        
        # Só guarda o bloco se tiver conteúdo real (mais de 50 caracteres)
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
            
        # O recuo da janela deslizante para criar a sobreposição
        start += chunk_size - overlap

    return chunks