import os
import sys

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

st.set_page_config(
    page_title="RAGForge Document Upload",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling immediately to prevent transition blink
from utils import apply_custom_styles
apply_custom_styles()

# Heavy/slow library imports
import time
import pandas as pd
from utils import render_sidebar, initialize_session_state, render_header
from utils import parse_pdf, parse_docx, parse_txt

# Initialize state and render layout elements
initialize_session_state()
render_sidebar()
render_header()

# Page Header
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem;">📄 Document Ingestion</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; max-width: 600px;">
        Upload files to build your knowledge base. Watch the step-by-step pipeline as text is extracted, chunked, and vectorized.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats Cards
stats = st.session_state.rag_pipeline.vector_store.get_collection_stats()
total_docs = stats.get("total_documents", 0)
total_chunks = stats.get("total_chunks", 0)
size_bytes = stats.get("collection_size_bytes", 0)
size_kb = size_bytes / 1024.0

vectorizer = st.session_state.rag_pipeline.vector_store.vectorizer
vector_dims = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Documents", total_docs)
with c2:
    st.metric("Chunks", total_chunks)
with c3:
    st.metric("TF-IDF Dims", vector_dims)
with c4:
    st.metric("DB Size", f"{size_kb:.1f} KB")

st.markdown('<div style="height:1px;background:rgba(99,102,241,0.06);margin:1.25rem 0;"></div>', unsafe_allow_html=True)

# Two Column Layout
col_left, col_right = st.columns([3, 2])

with col_left:
    with st.container(border=True):
        st.markdown("""
        <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.75rem;">
            Upload & Index
        </p>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files to ingest",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="PDF (.pdf), Word (.docx), Plain Text (.txt)"
        )
        
        ingest_button = st.button("🚀 Ingest Documents", use_container_width=True)
        
        if ingest_button and uploaded_files:
            success_count = 0
            total_new_chunks = 0
            
            overall_progress = st.progress(0)
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                
                with st.status(f"Processing `{file_name}`...", expanded=True) as status:
                    try:
                        status.write("⏳ Extracting text...")
                        time.sleep(0.2)
                        if file_name.endswith(".pdf"):
                            extracted_text = parse_pdf(file_bytes)
                        elif file_name.endswith(".docx"):
                            extracted_text = parse_docx(file_bytes)
                        else:
                            extracted_text = parse_txt(file_bytes)
                            
                        if not extracted_text.strip():
                            status.update(label=f"❌ `{file_name}`: No text found", state="error")
                            continue
                            
                        status.write("✅ Text extracted → ⏳ Chunking...")
                        time.sleep(0.2)
                        
                        vector_store = st.session_state.rag_pipeline.vector_store
                        new_chunks = vector_store.chunk_text(extracted_text)
                        
                        if not new_chunks:
                            status.update(label=f"⚠️ `{file_name}`: 0 chunks produced", state="warning")
                            continue
                            
                        status.write("✅ Chunked → ⏳ Vectorizing with TF-IDF...")
                        time.sleep(0.2)
                        
                        existing_data = vector_store.collection.get(include=["documents", "metadatas"])
                        existing_docs = existing_data.get("documents", [])
                        existing_metas = existing_data.get("metadatas", [])
                        
                        all_chunks = list(existing_docs) + new_chunks
                        all_metas = list(existing_metas)
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        for i, chunk in enumerate(new_chunks):
                            all_metas.append({
                                "source": file_name,
                                "chunk_index": i,
                                "timestamp": current_time
                            })
                            
                        vectors = vector_store.vectorize(all_chunks)
                        
                        status.write("✅ Vectorized → ⏳ Storing in ChromaDB...")
                        time.sleep(0.2)
                        
                        vector_store.store(all_chunks, vectors, all_metas)
                        
                        status.update(
                            label=f"✅ `{file_name}` — {len(new_chunks)} chunks indexed",
                            state="complete"
                        )
                        
                        total_new_chunks += len(new_chunks)
                        success_count += 1
                        
                    except Exception as e:
                        status.update(label=f"❌ Error: {str(e)}", state="error")
                
                overall_progress.progress((file_idx + 1) / len(uploaded_files))
                
            if success_count > 0:
                st.success(f"✅ Ingested {success_count} file(s) — {total_new_chunks} new chunks added!")
                st.session_state.ingested_docs = st.session_state.rag_pipeline.vector_store.get_document_sources()
                st.rerun()
        elif ingest_button:
            st.warning("Please upload files first.")

with col_right:
    with st.container(border=True):
        st.markdown("""
        <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.75rem;">
            Knowledge Base
        </p>
        """, unsafe_allow_html=True)
        
        data = st.session_state.rag_pipeline.vector_store.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        if not metadatas:
            st.markdown("""
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2.5rem 1.5rem;
                background: rgba(255, 255, 255, 0.01);
                border: 1px dashed rgba(99, 102, 241, 0.15);
                border-radius: 12px;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.6;">🗂️</div>
                <span style="font-size: 0.95rem; font-weight: 700; color: #cbd5e1; display: block; margin-bottom: 0.25rem;">Knowledge Base Empty</span>
                <span style="font-size: 0.78rem; color: #64748b; max-width: 240px; line-height: 1.5; display: block;">
                    Upload files on the left to start building your vector store.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            doc_stats = {}
            for meta in metadatas:
                if meta:
                    source = meta.get("source", "Unknown")
                    timestamp = meta.get("timestamp", "—")
                    if source not in doc_stats:
                        doc_stats[source] = {"Filename": source, "Chunks": 0, "Status": "✅ Active", "Indexed": timestamp}
                    doc_stats[source]["Chunks"] += 1
                    
            # Professional list view of indexed documents
            for source, stats in doc_stats.items():
                item_col1, item_col2 = st.columns([5, 1], vertical_alignment="center")
                with item_col1:
                    st.markdown(f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.02);
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        padding: 0.6rem 0.8rem;
                        border-radius: 8px;
                        margin-bottom: 0.5rem;
                    ">
                        <span style="font-size: 0.88rem; font-weight: 600; color: #f1f5f9; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">📄 {source}</span>
                        <span style="font-size: 0.7rem; color: #94a3b8;">{stats['Chunks']} chunks · {stats['Indexed']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with item_col2:
                    # Small, compact trash button with destructive styling
                    if st.button("🗑️", key=f"del_{source}", help=f"Remove {source}", use_container_width=True):
                        st.session_state.rag_pipeline.delete_document(source)
                        st.success(f"Removed `{source}`")
                        st.session_state.ingested_docs = st.session_state.rag_pipeline.vector_store.get_document_sources()
                        st.rerun()
                        
            st.markdown('<div style="height:1px;background:rgba(99,102,241,0.06);margin:1.25rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Entire Knowledge Base", key="clear_db_btn", use_container_width=True):
                st.session_state.confirm_clear = True
                
            if st.session_state.get("confirm_clear", False):
                st.warning("⚠️ This will permanently delete all indexed data.")
                y, n = st.columns(2)
                with y:
                    if st.button("Yes, Clear", key="btn_yes", use_container_width=True):
                        st.session_state.rag_pipeline.clear_database()
                        st.session_state.chat_history = []
                        st.session_state.ingested_docs = []
                        st.session_state.confirm_clear = False
                        st.success("Knowledge base cleared!")
                        st.rerun()
                with n:
                    if st.button("Cancel", key="btn_no", use_container_width=True):
                        st.session_state.confirm_clear = False
                        st.rerun()

# Verification Section
if total_chunks > 0:
    st.markdown('<div style="height:1px;background:rgba(99,102,241,0.06);margin:1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.75rem;">
        🔍 Ingestion Verification
    </p>
    """, unsafe_allow_html=True)
    
    test_query = st.text_input("Search your indexed documents", placeholder="Type keywords to verify...")
    if st.button("Verify Search") and test_query:
        results = st.session_state.rag_pipeline.vector_store.search(test_query, k=2)
        if not results:
            st.warning("No matches found.")
        else:
            st.success(f"Found {len(results)} matching chunks!")
            import html
            for idx, res in enumerate(results):
                source = res["metadata"].get("source", "Unknown")
                score = res.get("similarity_score", 0.0)
                pct = max(0.0, min(1.0, score)) * 100
                escaped_content = html.escape(res["page_content"])
                
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
