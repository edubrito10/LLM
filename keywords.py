import re
import json

_MAX_WORDS = 2000


def extract_keywords(text: str, llm, n: int = 15) -> list[str]:
    """
    Identifica as n palavras-chave ou expressões técnicas mais relevantes
    presentes no documento.

    Args:
        text: texto completo do documento
        llm:  modelo de linguagem (Ollama)
        n:    número de palavras-chave a extrair

    Returns:
        Lista de strings com as palavras-chave extraídas.
    """
    excerpt = _truncate(text, _MAX_WORDS)

    prompt = f"""Analisa o seguinte texto académico e extrai exatamente {n} \
palavras-chave ou expressões-chave técnicas mais relevantes. \
Responde APENAS com uma lista JSON de strings, sem mais nada, \
sem markdown, sem explicações. \
Exemplo de formato: ["machine learning", "redes neuronais", "processamento de linguagem"]

TEXTO:
{excerpt}

LISTA JSON:"""

    raw = llm.invoke(prompt).strip()
    return _parse_keywords(raw, n)


def _parse_keywords(raw: str, n: int) -> list[str]:
    """Tenta extrair uma lista de strings do output do LLM."""
    # Tentativa 1: JSON direto
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            kws = json.loads(match.group())
            return [k.strip() for k in kws if isinstance(k, str)][:n]
        except json.JSONDecodeError:
            pass

    # Tentativa 2: fallback — split por vírgulas ou newlines
    cleaned = re.sub(r'[\[\]"{}]', '', raw)
    return [k.strip() for k in re.split(r'[,\n]', cleaned) if k.strip()][:n]


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text
