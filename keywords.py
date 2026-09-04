import re
import json

# Limite de palavras para não sobrecarregar o contexto do modelo
_MAX_WORDS = 2000


def extract_keywords(text: str, llm, n: int = 15) -> list[str]:
    """
    Prepara o texto cortando-o pelo limite máximo e aplica Engenharia de Prompt.
    Força o LLM a comportar-se como uma API, exigindo que ele devolva 
    exclusivamente um formato de lista JSON. O facto de incluir um Exemplo
    no prompt ajuda o modelo a não cometer erros de formatação.
    """
    excerpt = _truncate(text, _MAX_WORDS)

    # Prompt rigoroso com proibição de conversas ou explicações
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
    """
    Mesmo pedindo só JSON, os LLMs às vezes respondem com texto extra.
    Esta função tenta extrair a lista à força em 2 etapas. Se a conversão 
    pura de JSON falhar, recorre a Expressões Regulares (Regex) para destruir 
    o lixo de formatação e separar as palavras pelas vírgulas.
    """
    # Tentativa 1: Procurar a lista (entre parênteses retos) e forçar o JSON
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            kws = json.loads(match.group())
            # Devolve a lista, garantindo que são strings e cortando no limite 'n'
            return [k.strip() for k in kws if isinstance(k, str)][:n]
        except json.JSONDecodeError:
            pass # Se o JSON estiver malformado, o código não para, avança para a Tentativa 2

    # Tentativa 2: Fallback (Plano B) agressivo via Expressões Regulares
    # Destrói parênteses, chaves e aspas, deixando só o texto puro
    cleaned = re.sub(r'[\[\]"{}]', '', raw)
    # Separa por vírgulas ou mudanças de linha e devolve a lista
    return [k.strip() for k in re.split(r'[,\n]', cleaned) if k.strip()][:n]


def _truncate(text: str, max_words: int) -> str:
    """
    Corta as palavras num limite fixo para evitar estourar a memória (Out of Memory).
    """
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text