"""
rag.py
------
Pipeline RAG (Retrieval-Augmented Generation) para responder a perguntas
sobre o documento carregado, com suporte a histórico de conversa.
"""

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

    # 2. Formatar histórico (últimas 4 trocas)
    history_str = _format_history(history[-8:])  # 8 msgs = 4 trocas

    # 3. Construir prompt
    prompt = f"""És um assistente académico especializado em análise de documentos. \
Responde em português europeu, de forma clara e fundamentada, \
com base EXCLUSIVAMENTE no contexto fornecido abaixo. \
Se a resposta não estiver no contexto, diz que não encontraste informação suficiente.

CONTEXTO DO DOCUMENTO:
{context}

HISTÓRICO DA CONVERSA:
{history_str}

PERGUNTA: {question}

RESPOSTA (em português europeu, fundamentada no contexto):"""

    answer = llm.invoke(prompt)
    return answer.strip(), chunks


def _format_history(messages: list[dict]) -> str:
    """Formata o histórico de mensagens para incluir no prompt."""
    lines = []
    for msg in messages:
        role = "Utilizador" if msg["role"] == "user" else "Assistente"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "(sem histórico)"
