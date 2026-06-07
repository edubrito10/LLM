from vectorstore import retrieve_context
 
 
def rag_answer(
    question: str,
    collection,
    embeddings,
    llm,
    history: list[dict],
    k: int = 4,
) -> tuple[str, list[str]]:
    """
    Responde a uma pergunta usando RAG sobre o documento indexado.
 
    O contexto é recuperado do vectorstore e incluído no prompt junto
    com as últimas trocas do histórico de conversa.
 
    Args:
        question:   pergunta do utilizador em linguagem natural
        collection: coleção ChromaDB com os chunks do documento
        embeddings: modelo de embeddings
        llm:        modelo de linguagem (Ollama)
        history:    histórico de mensagens [{"role": ..., "content": ...}]
        k:          número de chunks a recuperar
 
    Returns:
        Tuplo (resposta_str, lista_de_chunks_usados)
    """
    # 1. Recuperar contexto relevante
    chunks = retrieve_context(question, collection, embeddings, k=k)
    context = "\n\n---\n\n".join(chunks)
 
    # 2. Formatar histórico (últimas 3 trocas, resumido)
    history_str = _format_history(history[-6:])  # 6 msgs = 3 trocas
 
    # 3. Construir prompt
    prompt = f"""És um assistente académico especializado em análise de documentos.
Responde em português europeu, de forma clara e concisa, com base EXCLUSIVAMENTE no contexto do documento fornecido abaixo.
Se a resposta não estiver no contexto, diz apenas: "Não encontrei informação suficiente no documento para responder a essa pergunta."
Não repitas perguntas anteriores. Não incluas secções de FAQ. Responde diretamente à pergunta atual.
 
CONTEXTO DO DOCUMENTO:
{context}
 
HISTÓRICO RECENTE (apenas para referência de contexto):
{history_str}
 
PERGUNTA ATUAL: {question}
 
RESPOSTA DIRETA (em português europeu, máximo 3 parágrafos):"""
 
    answer = llm.invoke(prompt)
    return answer.strip(), chunks
 
 
def _format_history(messages: list[dict]) -> str:
    """
    Formata o histórico de mensagens para incluir no prompt.
    Trunca respostas longas do assistente para evitar contaminação do prompt.
    """
    if not messages:
        return "(sem histórico)"
 
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "").strip()
 
        if role == "user":
            lines.append(f"Utilizador: {content}")
        else:
            # Truncar respostas longas do assistente (max 200 chars)
            truncated = content[:200] + "…" if len(content) > 200 else content
            lines.append(f"Assistente: {truncated}")
 
    return "\n".join(lines)