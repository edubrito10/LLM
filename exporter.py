import time
 
 
def build_pdf_report(
    filename: str, summary: str, keywords: list[str], chat_history: list[dict],
) -> bytes:
    """
    Orquestra a criação do ficheiro PDF. Ativa quebras automáticas de página
    e chama os blocos de desenho (Helpers) um a um. 
    Retorna o ficheiro num formato de 'bytes' para que o Streamlit
    o consiga enviar diretamente para a pasta de Transferências do utilizador.
    """
    from fpdf import FPDF
 
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
 
    # Desenha o cabeçalho
    _add_title_block(pdf, filename)
 
    # Adiciona as secções apenas se elas existirem na memória da sessão
    if summary:
        _add_section(pdf, "Resumo Executivo", summary)
    if keywords:
        _add_keywords_section(pdf, keywords)
    if chat_history:
        _add_chat_section(pdf, chat_history)
 
    return bytes(pdf.output())
 
 
# ─── Helpers internos
 
def _safe(text: str) -> str:
    """
    O fpdf2 sem fontes embutidas crasha com Unicode complexo (ex: Emojis).
    Esta função força a conversão do texto para latin-1 (para manter os acentos PT)
    e substitui caracteres inválidos, evitando falhas críticas na exportação.
    """
    return text.encode("latin-1", errors="replace").decode("latin-1")
 
 
def _add_title_block(pdf, filename: str) -> None:
    # Formata Títulos (Helvetica, Negrito, Tamanho 20)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 12, "LocaLLM Docs - Relatorio de Analise", ln=True)
 
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Documento: {_safe(filename)}", ln=True)
    pdf.cell(0, 6, f"Gerado em: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(8)
 
    # Linha separadora cinzenta horizontal
    pdf.set_draw_color(200, 200, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
 
 
def _add_section(pdf, title: str, content: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, title, ln=True)
 
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    # multi_cell é usado para textos longos, pois faz quebras de linha automáticas
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
    """
    Escreve o histórico no PDF, formatando dinamicamente a cor da fonte
    consoante seja o utilizador a falar ou o modelo de Inteligência Artificial.
    """
    # Filtra mensagens que estejam vazias por segurança
    messages = [m for m in chat_history if m.get("content", "").strip()]
    if not messages:
        return
 
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 10, "Historico de Chat", ln=True)
 
    pdf.set_draw_color(200, 200, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
 
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        role_label = "Utilizador" if role == "user" else "Assistente"
 
        pdf.set_font("Helvetica", "B", 11)
        # Lógica de interface: cores de fonte diferentes para separar visualmente as entidades
        pdf.set_text_color(80, 60, 120 if role == "assistant" else 160)
        pdf.cell(0, 7, f"{role_label}:", ln=True)
 
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, _safe(content))
        pdf.ln(3)