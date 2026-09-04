# 🔍 LocaLLM Docs: Privacy-First RAG & Document Analysis

**Institution:** University of Beira Interior (UBI) - MSc in Computer Engineering  
**Author:** Eduardo Brito  

---

## 📌 Project Overview
**LocaLLM Docs** is an advanced, 100% offline document analysis system built on the Retrieval-Augmented Generation (RAG) paradigm. It allows users to upload complex PDF and Word documents, converse with their content via a natural language interface, and generate automated executive summaries—all without relying on external APIs.

This project was engineered with a strict focus on **data privacy** and **VRAM optimization** for consumer-grade GPUs (e.g., NVIDIA RTX 3060), achieving full RAG inference and context retrieval in under 15 seconds.

---

## 🚀 Key Engineering Features

* **Optimized Vector Pipeline:** Implements in-memory ChromaDB semantic indexing with precise document chunking (500 words with an 80-word overlap) to preserve context boundaries without exhausting GPU limits.
* **Smart Summarization & NLP:** Uses regex-based preprocessing to automatically strip bibliographies and AI declarations before summarization, preventing LLM hallucinations.
* **VRAM-Optimized Offline Translation:** Integrates a local MarianMT (EN→PT) translation module. It uses batch processing (batch_size = 5) and PyTorch's `torch.no_grad()` during inference to strictly manage memory limits.
* **Strict Prompt Engineering:** Extracts key concepts by forcing the LLM (Llama 3.2 3B) to output strictly formatted JSON arrays, backed by regex fallbacks.
* **PDF Reporting:** Generates comprehensive PDF reports of the session using `fpdf2`, handling complex text encodings dynamically.

---

## 🛠️ Tech Stack

* **Interface:** Streamlit
* **AI/LLM Engine:** Ollama (`llama3.2:3b`)
* **Embeddings:** `nomic-embed-text` (768 dimensions)
* **Vector Database:** ChromaDB
* **Frameworks:** LangChain, PyTorch, Transformers (Hugging Face)

---

## 💻 Setup & Execution

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com). Then, pull the required models:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text