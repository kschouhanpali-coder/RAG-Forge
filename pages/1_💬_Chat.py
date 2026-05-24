import os
import sys
import re

# Ensure project root is on Python path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)
os.environ["STREAMLIT_CONFIG_DIR"] = os.path.join(workspace_path, ".streamlit")

# Bypass clashing global modules on Streamlit Cloud
for clashing_module in ['utils', 'config']:
    if clashing_module in sys.modules:
        m = sys.modules[clashing_module]
        has_file = hasattr(m, '__file__') and m.__file__ is not None
        if not has_file or not m.__file__.startswith(workspace_path):
            del sys.modules[clashing_module]

import streamlit as st
import time
from utils import apply_custom_styles, render_sidebar, initialize_session_state, render_header
from config import DEFAULT_TOP_K

st.set_page_config(
    page_title="RAGForge Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()
initialize_session_state()
render_sidebar()
render_header()

# Page Header
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.25rem;">💬 Chat Laboratory</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; max-width: 600px;">
        Query your knowledge base and inspect the exact data chunks informing each response.
    </p>
</div>
""", unsafe_allow_html=True)



# RAG Parameter Panel
with st.expander("⚙️ Pipeline Configuration", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)
    with col2:
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with col3:
        st.markdown(
            '<p style="font-size: 0.82rem; color: #64748b; padding-top: 1.2rem; line-height: 1.5;">'
            'Adjust retrieval width and generation creativity in real-time.</p>',
            unsafe_allow_html=True
        )

# Update temperature
if "rag_pipeline" in st.session_state:
    st.session_state.rag_pipeline.llm.temperature = temperature

# Pipeline steps helper
def render_pipeline_steps(query_txt, metrics):
    with st.expander("🔄 Pipeline Trace", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Preprocessed Query**")
            st.code(metrics.get('preprocessed_query', ''), language="text")
        with c2:
            vocab_size = 0
            if hasattr(st.session_state.rag_pipeline.vector_store, 'vectorizer') and hasattr(st.session_state.rag_pipeline.vector_store.vectorizer, 'vocabulary_'):
                if st.session_state.rag_pipeline.vector_store.vectorizer.vocabulary_ is not None:
                    vocab_size = len(st.session_state.rag_pipeline.vector_store.vectorizer.vocabulary_)
            st.metric("TF-IDF Dimensions", vocab_size)
        with c3:
            st.metric("Chunks Searched", metrics.get('chunks_searched', 0))
        
        l1, l2, l3 = st.columns(3)
        with l1:
            st.metric("Retrieval Latency", f"{metrics.get('retrieval_latency', 0.0):.4f}s")
        with l2:
            st.metric("Generation Latency", f"{metrics.get('generation_latency', 0.0):.4f}s")
        with l3:
            st.metric("Total Latency", f"{metrics.get('total_latency', 0.0):.4f}s")

# Chat Container
chat_container = st.container()

# Render chat history
with chat_container:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            if role == "user":
                st.markdown('<p style="color: #818cf8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 6px;">👤 You</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: #f472b6; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 6px;">🤖 RAGForge</p>', unsafe_allow_html=True)
            st.markdown(content)
            
            if role == "assistant" and "sources" in message and message["sources"]:
                sources = message["sources"]
                metrics = message.get("metrics", {})
                
                with st.expander(f"📚 Retrieved Context ({len(sources)} chunks)", expanded=False):
                    import html
                    for idx, chunk in enumerate(sources):
                        source = chunk["metadata"].get("source", "Unknown")
                        score = chunk.get("similarity_score", 0.0)
                        pct = max(0.0, min(1.0, score)) * 100
                        escaped_content = html.escape(chunk["page_content"])
                        
                        card_html = f"""
                        <div class="rf-match-card">
                            <div class="rf-match-header">
                                <span class="rf-match-badge">Match {idx+1}</span>
                                <span class="rf-match-source" title="{source}">{source}</span>
                                <span class="rf-match-score">Score: {score:.4f}</span>
                            </div>
                            <div class="rf-match-bar-container">
                                <div class="rf-match-bar-fill" style="width: {pct}%;"></div>
                            </div>
                            <div class="rf-match-body">{escaped_content}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                
                render_pipeline_steps(message.get("query", ""), metrics)

# Chat Input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    with chat_container:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown('<p style="color: #818cf8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 6px;">👤 You</p>', unsafe_allow_html=True)
            st.markdown(user_query)
            
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if not st.session_state.rag_pipeline.llm.is_configured():
        err_msg = "⚠️ Please set your Gemini API key in the sidebar to start chatting."
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<p style="color: #f472b6; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 6px;">🤖 RAGForge</p>', unsafe_allow_html=True)
                st.markdown(err_msg)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": err_msg,
            "llm_mode": "System"
        })
    else:
        with st.spinner("🔍 Searching knowledge base & generating response..."):
            answer, sources, scores, metrics = st.session_state.rag_pipeline.query(user_query, top_k=top_k)
            
        if sources:
            metrics_log = metrics.copy()
            metrics_log["query"] = user_query
            metrics_log["timestamp"] = time.strftime("%H:%M:%S")
            st.session_state.metrics_history.append(metrics_log)
        
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown('<p style="color: #f472b6; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.4rem 0; display: flex; align-items: center; gap: 6px;">🤖 RAGForge</p>', unsafe_allow_html=True)
                st.markdown(answer)
                
                if sources:
                    with st.expander(f"📚 Retrieved Context ({len(sources)} chunks)", expanded=False):
                        import html
                        for idx, chunk in enumerate(sources):
                            source = chunk["metadata"].get("source", "Unknown")
                            score = chunk.get("similarity_score", 0.0)
                            pct = max(0.0, min(1.0, score)) * 100
                            escaped_content = html.escape(chunk["page_content"])
                            
                            card_html = f"""
                            <div class="rf-match-card">
                                <div class="rf-match-header">
                                    <span class="rf-match-badge">Match {idx+1}</span>
                                    <span class="rf-match-source" title="{source}">{source}</span>
                                    <span class="rf-match-score">Score: {score:.4f}</span>
                                </div>
                                <div class="rf-match-bar-container">
                                    <div class="rf-match-bar-fill" style="width: {pct}%;"></div>
                                </div>
                                <div class="rf-match-body">{escaped_content}</div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                    
                    render_pipeline_steps(user_query, metrics)
                        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "metrics": metrics,
            "query": user_query,
            "llm_mode": st.session_state.get("llm_mode", "Gemini API")
        })
        st.rerun()

# Clear button
if st.session_state.chat_history:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
