# 🔍 LocaLLM Docs — Setup & Execução

## Estrutura do Projeto
```
lupa_literaria/
├── app.py            ← entrada principal (Streamlit)
├── models.py         ← carregamento cacheado dos modelos
├── document.py       ← extração de texto (PDF + Word) e chunking
├── vectorstore.py    ← ChromaDB: indexação e retrieval
├── rag.py            ← pipeline RAG com histórico de chat
├── summarizer.py     ← sumarização automática
├── keywords.py       ← extração de palavras-chave
├── translator.py     ← tradução EN→PT (MarianMT offline)
├── exporter.py       ← geração de relatório PDF
├── requirements.txt
└── README.md
```

---

## 1. Instalar Ollama
Vai a https://ollama.com e instala para Windows.

Depois, num terminal:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---

## 2. Instalar dependências Python

### Opção A — CPU apenas
```bash
pip install -r requirements.txt
```

### Opção B — GPU NVIDIA (RTX 3060, recomendado)
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Download do modelo de tradução (uma vez, requer internet)
Na primeira execução, o modelo MarianMT (~300 MB) é descarregado
automaticamente e fica em cache em `~/.cache/huggingface/`.
Depois funciona completamente offline.

---

## 4. Executar a aplicação
```bash
# Terminal 1 — garante que o Ollama está a correr:
ollama serve

# Terminal 2 — inicia a aplicação:
streamlit run app.py
```

Abre no browser em: **http://localhost:8501**

---
