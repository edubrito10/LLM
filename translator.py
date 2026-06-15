import re


def translate_en_to_pt(text: str, tokenizer, model) -> str:
    """
    Traduz um texto de Inglês para Português Europeu.

    Divide o texto em frases para respeitar o limite de tokens do modelo
    e processa em batches para eficiência.

    Args:
        text:      texto em inglês a traduzir
        tokenizer: MarianTokenizer carregado
        model:     MarianMTModel carregado

    Returns:
        Texto traduzido para português.
    """
    import torch

    sentences = _split_sentences(text)
    if not sentences:
        return text

    translated = []
    batch_size = 5

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_length=512)
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translated.extend(decoded)

    return " ".join(translated)


def _split_sentences(text: str) -> list[str]:
    """Divide o texto em frases usando pontuação como delimitador"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
