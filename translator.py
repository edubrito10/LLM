import re


def translate_en_to_pt(text: str, tokenizer, model) -> str:
    """

    Traduz textos pesados sem encravar o computador.
    Usa o processamento em lotes (batch_size = 5) e, crucialmente,
    o torch.no_grad() para poupar memória da placa gráfica, já que
    estamos na fase de inferência e não de treino do modelo.
    """
    import torch

    # 1. Corta o texto grande em frases isoladas
    sentences = _split_sentences(text)
    if not sentences:
        return text

    translated = []
    # Processa 5 frases de cada vez para ser rápido e poupar memória
    batch_size = 5

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        
        # 2. Tokenização: converte as palavras em tensores matemáticos
        # Aplica truncatura aos 512 tokens para respeitar o limite do MarianMT
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # 3. Otimização crítica de PyTorch: Desliga o cálculo de gradientes
        with torch.no_grad():
            # Gera a tradução usando Beam Search (num_beams=4) para frases mais fluídas
            outputs = model.generate(**inputs, num_beams=4, max_length=512)
            
        # 4. Descodificação: transforma os tensores matemáticos de volta em texto PT
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translated.extend(decoded)

    return " ".join(translated)


def _split_sentences(text: str) -> list[str]:
    """
    Usa Expressões Regulares para encontrar a pontuação final (. ! ?)
    e cortar o texto em frases soltas. Impede que o modelo de tradução
    receba blocos de texto maiores do que a sua janela de contexto (512 tokens).
    """
    # Corta o texto no espaço vazio que aparece imediatamente após a pontuação
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Limpa espaços em branco perdidos e devolve a lista
    return [s.strip() for s in sentences if s.strip()]