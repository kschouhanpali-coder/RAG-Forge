import os
import sys

# Ensure project root is on Python path for module imports
workspace_path = os.path.dirname(os.path.abspath(__file__))
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
from utils import apply_custom_styles, render_sidebar, initialize_session_state, render_header

# Page configuration
st.set_page_config(
    page_title="RAGForge - Pipeline Lab",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling immediately to prevent transition blink
apply_custom_styles()

# Initialize global states
initialize_session_state()

# Render Sidebar
render_sidebar()

# Top-left branding header
render_header()

# Hero Section
st.markdown("""
<div style="margin-top: 0.5rem; margin-bottom: 2.5rem;">
    <p style="font-size: 1.15rem; color: #94a3b8; max-width: 720px; line-height: 1.7; margin: 0;">
        An interactive studio for designing, debugging, and analyzing 
        Retrieval-Augmented Generation workflows. Understand exactly how text preprocessing, 
        vector indexing, TF-IDF similarity search, and LLM generation work in tandem.
    </p>
</div>
""", unsafe_allow_html=True)

# Pipeline Flow Diagram
st.markdown("""
<div style="margin-bottom: 2.5rem;">
    <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 1rem;">
        Pipeline Architecture
    </p>
    <div class="rf-pipeline-container">
        <div class="rf-pipeline-node">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">📝</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em;">Query</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">User input</div>
        </div>
        <div class="rf-pipeline-arrow">→</div>
        <div class="rf-pipeline-node">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">🧹</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em;">Clean</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">Preprocess</div>
        </div>
        <div class="rf-pipeline-arrow">→</div>
        <div class="rf-pipeline-node">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">📊</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em;">TF-IDF</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">Vectorize</div>
        </div>
        <div class="rf-pipeline-arrow">→</div>
        <div class="rf-pipeline-node">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">🗄️</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em;">Retrieve</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">ChromaDB</div>
        </div>
        <div class="rf-pipeline-arrow">→</div>
        <div class="rf-pipeline-node">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">📐</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.02em;">Rank</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">Cosine Sim</div>
        </div>
        <div class="rf-pipeline-arrow">→</div>
        <div class="rf-pipeline-node gemini">
            <div style="font-size: 1.2rem; margin-bottom: 0.25rem;">✨</div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #f472b6; letter-spacing: 0.02em;">Gemini</div>
            <div style="font-size: 0.62rem; color: #64748b; margin-top: 2px;">Generate</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="rf-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)); display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">💬</div>
            <span style="font-size: 1rem; font-weight: 700; color: #f1f5f9;">Interactive Chat</span>
        </div>
        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            Converse with your corpus in real-time. Tweak retrieval parameters and 
            inspect similarity scores for each retrieved chunk.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="rf-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(6,182,212,0.15)); display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">🔗</div>
            <span style="font-size: 1rem; font-weight: 700; color: #f1f5f9;">URL Scraper</span>
        </div>
        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            Extract and index content from web pages. Strips templates, scripts, 
            and feeds clean text into the vector store.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="rf-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(236,72,153,0.2), rgba(168,85,247,0.15)); display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">📄</div>
            <span style="font-size: 1rem; font-weight: 700; color: #f1f5f9;">Doc Uploader</span>
        </div>
        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            Ingest PDFs, Word documents, and text files. See real-time pipeline 
            progress as text is chunked and vectorized.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="rf-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, rgba(251,146,60,0.2), rgba(245,158,11,0.15)); display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">📊</div>
            <span style="font-size: 1rem; font-weight: 700; color: #f1f5f9;">RAG Analytics</span>
        </div>
        <p style="margin: 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            Audit pipeline performance with interactive charts. Monitor similarity 
            scores, latency, faithfulness and recall metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick Start
st.markdown("""
<div style="margin-top: 1rem;">
    <p style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 1rem;">
        Quick Start
    </p>
    <div style="
        background: rgba(15, 18, 33, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
    ">
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; font-size: 0.7rem; font-weight: 800; width: 22px; height: 22px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">1</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;">Enter your <strong style="color:#a5b4fc;">Gemini API Key</strong> in the sidebar</span>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; font-size: 0.7rem; font-weight: 800; width: 22px; height: 22px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">2</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;">Upload docs or scrape URLs to build your <strong style="color:#a5b4fc;">knowledge base</strong></span>
            </div>
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; font-size: 0.7rem; font-weight: 800; width: 22px; height: 22px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">3</span>
                <span style="font-size: 0.85rem; color: #cbd5e1;">Open <strong style="color:#a5b4fc;">Chat</strong> and start asking questions</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


