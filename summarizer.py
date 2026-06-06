"""
summarizer.py
-------------
Geração de resumos executivos do documento carregado,
com extensão ajustável (curto / médio / longo).
"""

# Número máximo de palavras do documento a enviar ao LLM
_MAX_WORDS = 3000

_LENGTH_DESCRIPTIONS = {
    "curto": "3 a 4 frases concisas que capturem a essência do documento",
    "médio": "2 a 3 parágrafos bem estruturados, cobrindo objetivos, metodologia e conclusões",
    "longo": "5 a 6 parágrafos detalhados, com secções temáticas claramente identificadas",
}


def summarize(text: str, llm, length: str = "médio") -> str:
    """
    Gera um resumo executivo do documento em português europeu.

    Args:
        text:   texto completo do documento
        llm:    modelo de linguagem (Ollama)
        length: extensão desejada — "curto", "médio" ou "longo"

    Returns:
        Resumo gerado como string.
    """
    detail = _LENGTH_DESCRIPTIONS.get(length, _LENGTH_DESCRIPTIONS["médio"])
    excerpt = _truncate(text, _MAX_WORDS)

    prompt = f"""És um especialista em síntese académica. \
Com base no seguinte texto, gera um resumo executivo em português europeu \
com {detail}. O resumo deve capturar os objetivos principais, \
a metodologia utilizada, os resultados obtidos e as conclusões do documento. \
Não inventes informação que não esteja no texto.

TEXTO:
{excerpt}

RESUMO EXECUTIVO:"""

    return llm.invoke(prompt).strip()


def _truncate(text: str, max_words: int) -> str:
    """Limita o texto a um número máximo de palavras."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[... texto truncado para processamento ...]"
