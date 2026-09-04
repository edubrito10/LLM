from vectorstore import retrieve_context
 
 
def rag_answer(
    question: str, collection, embeddings, llm, history: list[dict], k: int = 4,
) -> tuple[str, list[str]]:
    """
    Junta: o Contexto do documento, o Histórico da conversa e a Pergunta atual. 
    Depois, aplica regras rígidas (Prompt) para forçar o modelo a ser factual e a não alucinar respostas.
    """
    # 1. Vai à base de dados buscar os blocos de texto (chunks) relevantes
    chunks = retrieve_context(question, collection, embeddings, k=k)
    context = "\n\n---\n\n".join(chunks)
 
    # 2. Formata o histórico, mas apenas as últimas 6 mensagens (3 perguntas e 3 respostas)
    history_str = _format_history(history[-6:])
 
    # 3. O 'System Prompt' - As regras do jogo. 
    # É aqui que mitigas as alucinações ("Se não souberes, não inventes").
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
 
    # É aqui que a pergunta sai da interface (app.py) e vai para o motor (rag.py) pensar e gerar a resposta.
    answer = llm.invoke(prompt)
    return answer.strip(), chunks
 
 
def _format_history(messages: list[dict]) -> str:
    """
    Converte as mensagens do chat num formato legível para o LLM.
    O detalhe vital: as respostas anteriores do IA são truncadas (cortadas)
    aos 200 caracteres para evitar que o LLM se perca a ler os seus 
    próprios textos longos e se esqueça de responder à nova pergunta.
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
            # Se for a IA a falar, corta o texto aos 200 caracteres (Truncatura)
            truncated = content[:200] + "…" if len(content) > 200 else content
            lines.append(f"Assistente: {truncated}")
 
    return "\n".join(lines)