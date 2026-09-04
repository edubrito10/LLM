# Limite de segurança para não esgotar a memória/contexto do LLM
_MAX_WORDS = 3000
 
# Dicionário que dita o comportamento do LLM dependendo do tamanho escolhido
_LENGTH_DESCRIPTIONS = {
    "curto": "3 a 4 frases concisas que capturem a essência do documento",
    "médio": "2 a 3 parágrafos bem estruturados, cobrindo objetivos, metodologia e conclusões",
    "longo": "5 a 6 parágrafos detalhados, com secções temáticas claramente identificadas",
}
 
 
def summarize(text: str, llm, length: str = "médio") -> str:
    """
    Esta função prepara o 'terreno'. Vai buscar o nível de detalhe exigido,
    limpa a bibliografia do texto original, garante que o tamanho não excede
    o limite da placa gráfica e cria um Prompt rigoroso para forçar o LLM
    a fazer um resumo focado e sem invenções.
    """
    detail = _LENGTH_DESCRIPTIONS.get(length, _LENGTH_DESCRIPTIONS["médio"])
    
    # 1. Limpa o lixo (Referências)
    clean  = _remove_references(text)
    
    # 2. Garante que não passa das 3000 palavras
    excerpt = _truncate(clean, _MAX_WORDS)
 
    # 3. O 'System Prompt' super restritivo para garantir qualidade
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
 
    # 4. Devolve a resposta final limpa
    return llm.invoke(prompt).strip()
 
 
def _remove_references(text: str) -> str:
    """
    Usa Expressões Regulares para procurar a secção da Bibliografia.
    Assim que encontra a palavra "Referências" ou "References", corta 
    literalmente o documento a meio e descarta o resto. Impede que o LLM
    perca o seu tempo e tokens a tentar resumir fontes de pesquisa.
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
        # Se encontrou um destes cabeçalhos, corta o texto desde o início até aí
        if match:
            text = text[:match.start()]
            
    return text.strip()
 
 
def _truncate(text: str, max_words: int) -> str:
    """
    Se o documento (mesmo depois de limpo) for demasiado longo, 
    corta o texto para caber nos limites lógicos do modelo de IA local,
    anexando uma nota para indicar que houve corte.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[... texto truncado para processamento ...]"