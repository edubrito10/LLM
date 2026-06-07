import streamlit as st


@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str = "nomic-embed-text"):
    """Carrega o modelo de embeddings via Ollama."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model_name)


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "llama3.2:3b"):
    """Carrega o LLM via Ollama."""
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=model_name, temperature=0.2)


@st.cache_resource(show_spinner=False)
def load_translation_model():
    """
    Carrega o modelo de tradução Helsinki-NLP EN→PT (MarianMT).
    Na primeira execução descarrega ~300MB; depois funciona offline.
    Devolve (tokenizer, model) ou (None, mensagem_de_erro).
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        name = "Helsinki-NLP/opus-mt-en-pt"
        tokenizer = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        return tokenizer, model
    except Exception as e:
        return None, str(e)


def check_ollama() -> bool:
    """Verifica se o servidor Ollama está acessível."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False
