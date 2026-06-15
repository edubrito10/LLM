import time
import streamlit as st

# ─── PAGE CONFIG (tem de ser o primeiro comando Streamlit) ────────────────────
st.set_page_config(
    page_title="LocaLLM Docs",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# modulos locais
from models import load_embeddings, load_llm, load_translation_model, check_ollama
from document import extract_text, chunk_text
from vectorstore import build_vectorstore
from rag import rag_answer
from summarizer import summarize
from keywords import extract_keywords
from translator import translate_en_to_pt
from exporter import build_pdf_report

# style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&family=Lato:wght@300;400;700&display=swap');

:root {
    --bg-primary: #0f0e17;
    --bg-secondary: #1a1827;
    --bg-card: #1e1c2e;
    --accent-gold: #e8c87a;
    --accent-amber: #f5a623;
    --accent-rose: #e07a8f;
    --text-primary: #fffffe;
    --text-muted: #a7a9be;
    --border: #2e2b45;
    --success: #72efdd;
}

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.lupa-header { text-align: center; padding: 2rem 0 1rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.lupa-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-amber), var(--accent-rose));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; letter-spacing: -0.02em;
}
.lupa-subtitle {
    font-family: 'Lato', sans-serif; font-weight: 300;
    color: var(--text-muted); font-size: 1rem;
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.3rem;
}

.feature-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem; }
.feature-card h4 { font-family: 'Playfair Display', serif; color: var(--accent-gold); margin-top: 0; font-size: 1.1rem; }

.chat-user {
    background: linear-gradient(135deg, #2a2540, #1e1c2e);
    border-left: 3px solid var(--accent-amber);
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: 0.7rem 0; font-size: 0.95rem;
}
.chat-assistant {
    background: linear-gradient(135deg, #1a2535, #151e2e);
    border-left: 3px solid var(--success);
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: 0.7rem 0; font-size: 0.95rem; line-height: 1.7;
}
.chat-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem; }
.label-user { color: var(--accent-amber); }
.label-assistant { color: var(--success); }

.source-snippet {
    background: #12111e; border: 1px solid var(--border); border-radius: 8px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0;
    font-family: 'Source Code Pro', monospace; font-size: 0.78rem; color: var(--text-muted);
}

.kw-pill {
    display: inline-block;
    background: linear-gradient(135deg, #2a2040, #1e1c30);
    border: 1px solid var(--accent-gold); color: var(--accent-gold);
    border-radius: 20px; padding: 0.3rem 0.8rem; margin: 0.2rem;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.04em;
}

.status-ok   { color: var(--success);       font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; }
.status-warn { color: var(--accent-amber);  font-size: 0.8rem; }

div[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-amber));
    color: #0f0e17; font-weight: 700; border: none; border-radius: 8px;
    letter-spacing: 0.05em; transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
}
div[data-testid="stExpander"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="lupa-header">
    <h1 class="lupa-title">🔍 LocaLLM Docs</h1>
    <p class="lupa-subtitle">Sistema Avançado de Análise RAG Local · Offline</p>
</div>
""", unsafe_allow_html=True)

# session state
for key, default in {
    "doc_text": None,
    "doc_filename": None,
    "collection": None,
    "chat_history": [],
    "summary": None,
    "keywords": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuração")

    ollama_model = st.selectbox(
        "Modelo LLM (Ollama)",
        ["llama3.2:3b", "llama3.2:1b", "mistral:7b", "gemma2:2b"],
        help="Certifica-te que o modelo está instalado: ollama pull <modelo>",
    )

    st.markdown("---")
    st.markdown("### 📄 Carregar Documento")

    uploaded_file = st.file_uploader(
        "PDF ou Word (.docx)",
        type=["pdf", "docx"],
        help="O documento é processado localmente, sem enviar dados para qualquer servidor.",
    )

    chunk_size = st.slider("Tamanho do chunk (palavras)", 200, 800, 500, 50)
    top_k      = st.slider("Chunks relevantes para RAG (k)", 2, 8, 6)

    process_btn = st.button("🚀 Processar Documento", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌐 Tradução (EN→PT)")
    translate_enabled = st.toggle("Ativar tradução automática", value=False)

    st.markdown("---")
    st.markdown("### 📊 Estado do Sistema")

    if check_ollama():
        st.markdown('<span class="status-ok">✅ Ollama online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">⚠️ Ollama offline — corre: `ollama serve`</span>', unsafe_allow_html=True)

    if st.session_state.doc_filename:
        st.markdown(f'<span class="status-ok">📄 {st.session_state.doc_filename}</span>', unsafe_allow_html=True)
        n_turns = len(st.session_state.chat_history) // 2
        st.markdown(f'<span class="status-ok">✅ {n_turns} perguntas feitas</span>', unsafe_allow_html=True)

# processar documento
if process_btn and uploaded_file:
    with st.spinner("A processar documento… Por favor aguarda."):
        try:
            file_bytes = uploaded_file.read()
            filename   = uploaded_file.name

            text = extract_text(file_bytes, filename)

            if not text.strip():
                st.error("Não foi possível extrair texto do documento.")
                st.stop()

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=80)
            st.info(f"📦 {len(chunks)} chunks criados a partir de {len(text.split())} palavras.")

            embeddings = load_embeddings()
            collection = build_vectorstore(chunks, embeddings)

            st.session_state.doc_text     = text
            st.session_state.doc_filename = filename
            st.session_state.collection   = collection
            st.session_state.chat_history = []
            st.session_state.summary      = None
            st.session_state.keywords     = []

            st.success(f"✅ '{filename}' processado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar documento: {e}")

# tabs principais
if st.session_state.doc_text:
    tab_chat, tab_summary, tab_keywords, tab_export = st.tabs([
        "💬 Chat RAG", "📋 Resumo", "🏷️ Palavras-Chave", "📥 Exportar",
    ])

    # chat
    with tab_chat:
        st.markdown("#### 💬 Dialoga com o Documento")
        st.markdown(f'<span class="status-ok">📄 {st.session_state.doc_filename}</span>', unsafe_allow_html=True)
        st.markdown("")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label label-user">👤 Utilizador</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-assistant">
                    <div class="chat-label label-assistant">🤖 LocaLLM Docs</div>
                    {msg["content"]}
                </div>""", unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📎 Fontes usadas na resposta"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f'<div class="source-snippet"><b>Fonte {i}:</b> {src[:300]}…</div>',
                                unsafe_allow_html=True,
                            )

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "A tua pergunta",
                placeholder="Ex: Qual é o objetivo principal deste artigo? Que metodologia foi usada?",
                height=80,
                label_visibility="collapsed",
            )
            col1, col2 = st.columns([4, 1])
            with col2:
                submitted = st.form_submit_button("Enviar ➤", use_container_width=True)

        if submitted and user_input.strip():
            with st.spinner("A pensar…"):
                try:
                    embeddings = load_embeddings()
                    llm        = load_llm(ollama_model)

                    answer, sources = rag_answer(
                        user_input,
                        st.session_state.collection,
                        embeddings,
                        llm,
                        st.session_state.chat_history,
                        k=top_k,
                    )

                    if translate_enabled:
                        tokenizer, model = load_translation_model()
                        if tokenizer:
                            answer = translate_en_to_pt(answer, tokenizer, model)

                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro no RAG: {e}")

        if st.session_state.chat_history:
            if st.button("🗑️ Limpar conversa"):
                st.session_state.chat_history = []
                st.rerun()

    # resumo
    with tab_summary:
        st.markdown("#### 📋 Sumarização Automática")

        length_opt = st.radio("Extensão do resumo", ["curto", "médio", "longo"], index=1, horizontal=True)

        if st.button("⚡ Gerar Resumo"):
            with st.spinner("A sumarizar…"):
                try:
                    llm = load_llm(ollama_model)
                    result = summarize(st.session_state.doc_text, llm, length_opt)

                    if translate_enabled:
                        tokenizer, model = load_translation_model()
                        if tokenizer:
                            result = translate_en_to_pt(result, tokenizer, model)

                    st.session_state.summary = result
                except Exception as e:
                    st.error(f"Erro na sumarização: {e}")

        if st.session_state.summary:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown(f"**Resumo ({length_opt}):**")
            st.markdown(st.session_state.summary)
            st.markdown('</div>', unsafe_allow_html=True)

    # keywords
    with tab_keywords:
        st.markdown("#### 🏷️ Extração de Palavras-Chave")

        n_kws = st.slider("Número de palavras-chave", 5, 25, 15)

        if st.button("🔎 Extrair Palavras-Chave"):
            with st.spinner("A extrair conceitos-chave…"):
                try:
                    llm = load_llm(ollama_model)
                    kws = extract_keywords(st.session_state.doc_text, llm, n_kws)
                    st.session_state.keywords = kws
                except Exception as e:
                    st.error(f"Erro na extração: {e}")

        if st.session_state.keywords:
            pills = "".join(f'<span class="kw-pill">{kw}</span>' for kw in st.session_state.keywords)
            st.markdown(f'<div class="feature-card">{pills}</div>', unsafe_allow_html=True)

    # exportar
    with tab_export:
        st.markdown("#### 📥 Exportar Relatório de Análise")
        st.markdown("Gera um PDF com o resumo, palavras-chave e histórico de chat.")

        has_content = any([st.session_state.summary, st.session_state.keywords, st.session_state.chat_history])

        if not has_content:
            st.info("Gera primeiro um resumo ou faz perguntas ao documento para exportar resultados.")
        else:
            if st.button("📄 Gerar e Descarregar PDF"):
                with st.spinner("A gerar relatório PDF…"):
                    try:
                        pdf_bytes = build_pdf_report(
                            filename     = st.session_state.doc_filename or "documento",
                            summary      = st.session_state.summary or "",
                            keywords     = st.session_state.keywords,
                            chat_history = st.session_state.chat_history,
                        )
                        st.download_button(
                            label        = "⬇️ Descarregar Relatório PDF",
                            data         = pdf_bytes,
                            file_name    = f"lupa_relatorio_{time.strftime('%Y%m%d_%H%M')}.pdf",
                            mime         = "application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Erro ao gerar PDF: {e}")

# empty state
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; color: #a7a9be;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
        <h3 style="font-family: 'Playfair Display', serif; color: #e8c87a;">Nenhum documento carregado</h3>
        <p>Utiliza o painel lateral para carregar um ficheiro <strong>PDF</strong> ou <strong>Word (.docx)</strong><br>
        e clica em <strong>"Processar Documento"</strong> para começar.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("💬", "Chat RAG",        "Faz perguntas em linguagem natural diretamente ao documento."),
        ("📋", "Sumarização",     "Resumo executivo ajustável em curto, médio ou longo formato."),
        ("🏷️", "Palavras-Chave", "Extração automática dos conceitos mais relevantes."),
        ("📥", "Exportação",      "Relatório PDF completo com toda a análise gerada."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <h4>{title}</h4>
                <p style="color:#a7a9be; font-size:0.85rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
