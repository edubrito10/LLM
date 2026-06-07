import time
 
 
def build_pdf_report(
    filename: str,
    summary: str,
    keywords: list[str],
    chat_history: list[dict],
) -> bytes:
    """
    Gera um relatório PDF completo com a análise do documento.
 
    Args:
        filename:     nome do ficheiro original analisado
        summary:      resumo executivo gerado
        keywords:     lista de palavras-chave extraídas
        chat_history: lista de mensagens [{"role": ..., "content": ...}]
 
    Returns:
        Conteúdo do PDF em bytes, pronto para download.
    """
    from fpdf import FPDF
 
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
 
    _add_title_block(pdf, filename)
 
    if summary:
        _add_section(pdf, "Resumo Executivo", summary)
 
    if keywords:
        _add_keywords_section(pdf, keywords)
 
    if chat_history:
        _add_chat_section(pdf, chat_history)
 
    return bytes(pdf.output())
 
 
# ─── Helpers internos ─────────────────────────────────────────────────────────
 
def _safe(text: str) -> str:
    """Converte texto para latin-1 de forma segura (fpdf2 sem unicode)."""
    return text.encode("latin-1", errors="replace").decode("latin-1")
 
 
def _add_title_block(pdf, filename: str) -> None:
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 12, "LupaLiteraria - Relatorio de Analise", ln=True)
 
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Documento: {_safe(filename)}", ln=True)
    pdf.cell(0, 6, f"Gerado em: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(8)
 
    # Linha separadora
    pdf.set_draw_color(200, 200, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
 
 
def _add_section(pdf, title: str, content: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, title, ln=True)
 
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 7, _safe(content))
    pdf.ln(5)
 
 
def _add_keywords_section(pdf, keywords: list[str]) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, "Palavras-Chave", ln=True)
 
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    kw_line = " | ".join(keywords)
    pdf.multi_cell(0, 7, _safe(kw_line))
    pdf.ln(5)
 
 
def _add_chat_section(pdf, chat_history: list[dict]) -> None:
    # Filtrar apenas mensagens com conteúdo de texto (ignorar entradas vazias)
    messages = [m for m in chat_history if m.get("content", "").strip()]
    if not messages:
        return
 
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, "Historico de Chat", ln=True)
 
    # Linha separadora
    pdf.set_draw_color(200, 200, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
 
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        role_label = "Utilizador" if role == "user" else "Assistente"
 
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(80, 60, 120 if role == "assistant" else 160)
        pdf.cell(0, 7, f"{role_label}:", ln=True)
 
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, _safe(content))
        pdf.ln(3)