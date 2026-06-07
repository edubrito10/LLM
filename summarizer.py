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
    clean  = _remove_references(text)
    excerpt = _truncate(clean, _MAX_WORDS)
 
    prompt = f"""És um especialista em síntese académica. \
Com base no seguinte texto, gera um resumo executivo em português europeu \
com {detail}. O resumo deve capturar os objetivos principais, \
a metodologia utilizada, os resultados obtidos e as conclusões do documento. \
Não inventes informação que não esteja no texto. \
Não incluas referências bibliográficas, citações, notas de rodapé \
nem declarações de uso de IA no resumo.
 
TEXTO:
{excerpt}
 
RESUMO EXECUTIVO:"""
 
    return llm.invoke(prompt).strip()
 
 
def _remove_references(text: str) -> str:
    """
    Remove secções de referências bibliográficas, declarações de uso de IA
    e notas de rodapé do texto antes de o enviar ao LLM.
    """
    import re
    # Padrões comuns de início de secção de referências (PT e EN)
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*Referências\s*\n',
        r'\n\s*Bibliografia\s*\n',
        r'\n\s*Declaração de Uso de IA\s*\n',
        r'\n\s*Declaration of AI Use\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
    return text.strip()
 
 
def _truncate(text: str, max_words: int) -> str:
    """Limita o texto a um número máximo de palavras."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[... texto truncado para processamento ...]"