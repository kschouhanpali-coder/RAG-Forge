import os
import streamlit as st
from config import CUSTOM_CSS, LOGO_HTML

def save_api_key_to_env(new_key: str):
    """Saves the API key persistently into the .env file in the project root."""
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        lines = []
        updated = False
        
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                if line.strip().startswith("GOOGLE_API_KEY="):
                    lines[idx] = f'GOOGLE_API_KEY="{new_key}"\n'
                    updated = True
                    break
                    
        if not updated:
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append(f'GOOGLE_API_KEY="{new_key}"\n')
            
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error saving API key to .env: {e}")

def apply_custom_styles():
    """Applies the custom premium CSS styling to the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header():
    """Renders the RAGForge branding header at the top-left corner of the main content area."""
    st.markdown("""
    <div style="
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.25rem; padding-bottom: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.06);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <svg width="36" height="36" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0; filter: drop-shadow(0px 4px 12px rgba(99, 102, 241, 0.45));">
                <defs>
                    <linearGradient id="logoGradHeader" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#6366f1" />
                        <stop offset="50%" stop-color="#8b5cf6" />
                        <stop offset="100%" stop-color="#ec4899" />
                    </linearGradient>
                    <filter id="glowHeader" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>
                <rect x="2" y="2" width="96" height="96" rx="22" fill="#0b0d19" stroke="url(#logoGradHeader)" stroke-width="3"/>
                <circle cx="50" cy="50" r="32" stroke="url(#logoGradHeader)" stroke-width="2" stroke-dasharray="6 8" />
                <line x1="32" y1="50" x2="52" y2="34" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
                <line x1="32" y1="50" x2="52" y2="66" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
                <line x1="52" y1="34" x2="70" y2="50" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
                <line x1="52" y1="66" x2="70" y2="50" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
                <line x1="52" y1="34" x2="52" y2="66" stroke="url(#logoGradHeader)" stroke-width="1.5" stroke-dasharray="3 3"/>
                <circle cx="32" cy="50" r="5" fill="#f1f5f9"/>
                <circle cx="52" cy="34" r="5" fill="#818cf8" filter="url(#glowHeader)"/>
                <circle cx="52" cy="66" r="5" fill="#f472b6" filter="url(#glowHeader)"/>
                <circle cx="70" cy="50" r="6" fill="url(#logoGradHeader)" filter="url(#glowHeader)"/>
            </svg>
            <div>
                <span style="
                    font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 800;
                    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #cbd5e1 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -0.03em;
                ">RAGForge</span>
                <div style="
                    font-size: 0.65rem; color: #8b5cf6; font-weight: 700;
                    letter-spacing: 0.14em; text-transform: uppercase; margin-top: -3px;
                ">Pipeline Laboratory</div>
            </div>
        </div>
        <div style="
            font-size: 0.68rem; color: #475569; font-weight: 500;
            background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.1);
            padding: 0.3rem 0.75rem; border-radius: 20px;
        ">v1.0.0</div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Renders the common sidebar layout across all application pages."""
    with st.sidebar:
        # Logo
        st.markdown(LOGO_HTML, unsafe_allow_html=True)
        
        # Custom Page Navigation
        st.page_link("app.py", label="Dashboard", icon="⚙️")
        st.page_link("pages/1_💬_Chat.py", label="Chat", icon="💬")
        st.page_link("pages/2_📄_Document_Upload.py", label="Document Upload", icon="📄")
        st.page_link("pages/3_🔗_URL_Ingestion.py", label="URL Ingestion", icon="🔗")
        st.page_link("pages/4_📊_RAG_Metrics.py", label="RAG Metrics", icon="📊")
        
        # Divider
        st.markdown(
            '<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent); margin: 1rem 0 1.25rem 0;"></div>',
            unsafe_allow_html=True
        )
        
        # API Key Input
        saved_key = st.session_state.get("api_key", os.getenv("GOOGLE_API_KEY", ""))
        api_key_input = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            value=saved_key,
            placeholder="Paste your API key here...",
            help="Get your key from Google AI Studio (aistudio.google.com)"
        )
        
        if api_key_input != saved_key:
            st.session_state.api_key = api_key_input
            os.environ["GOOGLE_API_KEY"] = api_key_input
            save_api_key_to_env(api_key_input)
            if "llm_api_warning" in st.session_state:
                del st.session_state.llm_api_warning
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.update_api_key(api_key_input)
                
        # Determine LLM mode automatically
        has_key = bool(st.session_state.get("api_key") or os.getenv("GOOGLE_API_KEY"))
        st.session_state.llm_mode = "Gemini API" if has_key else "Mock LLM (Offline / No Key)"
                
        # Status badge
        if "rag_pipeline" in st.session_state:
            llm_mode = st.session_state.get("llm_mode", "Gemini API")
            if llm_mode == "Mock LLM (Offline / No Key)":
                st.markdown(
                    '<div style="'
                    'background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.18);'
                    'border-radius: 10px; padding: 0.5rem 0.75rem;'
                    'display: flex; align-items: center; gap: 8px; margin-top: 0.6rem;">'
                    '<div style="width:7px;height:7px;border-radius:50%;background:#818cf8;'
                    'box-shadow: 0 0 6px rgba(129,140,248,0.5);"></div>'
                    '<span style="color:#818cf8; font-weight:600; font-size:0.78rem;">Mock LLM Active (Offline)</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                is_llm_ready = st.session_state.rag_pipeline.llm.is_configured()
                if is_llm_ready:
                    st.markdown(
                        '<div style="'
                        'background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.18);'
                        'border-radius: 10px; padding: 0.5rem 0.75rem;'
                        'display: flex; align-items: center; gap: 8px; margin-top: 0.6rem;">'
                        '<div style="width:7px;height:7px;border-radius:50%;background:#34d399;'
                        'box-shadow: 0 0 6px rgba(52,211,153,0.5);"></div>'
                        '<span style="color:#34d399; font-weight:600; font-size:0.78rem;">API Connected</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="'
                        'background: rgba(251,146,60,0.06); border: 1px solid rgba(251,146,60,0.18);'
                        'border-radius: 10px; padding: 0.5rem 0.75rem;'
                        'display: flex; align-items: center; gap: 8px; margin-top: 0.6rem;">'
                        '<div style="width:7px;height:7px;border-radius:50%;background:#fb923c;"></div>'
                        '<span style="color:#fb923c; font-weight:600; font-size:0.78rem;">API Key Required</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                
            # Knowledge Index Section
            st.markdown(
                '<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent); margin: 1.25rem 0;"></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<p style="font-size: 0.65rem; color: #475569; text-transform: uppercase; '
                'letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.6rem;">Knowledge Index</p>',
                unsafe_allow_html=True
            )
            
            all_chunks = st.session_state.rag_pipeline.vector_store.get_all_chunks()
            doc_names = st.session_state.rag_pipeline.vector_store.get_document_sources()
            chunk_count = len(all_chunks.get("documents", []))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f'<div style="background: rgba(99,102,241,0.05); border: 1px solid rgba(99,102,241,0.1);'
                    f'border-radius: 10px; padding: 0.55rem; text-align: center;">'
                    f'<div style="font-size: 1.3rem; font-weight: 800; color: #818cf8; font-family: Outfit, sans-serif;">{len(doc_names)}</div>'
                    f'<div style="font-size: 0.6rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">Sources</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div style="background: rgba(236,72,153,0.05); border: 1px solid rgba(236,72,153,0.1);'
                    f'border-radius: 10px; padding: 0.55rem; text-align: center;">'
                    f'<div style="font-size: 1.3rem; font-weight: 800; color: #f472b6; font-family: Outfit, sans-serif;">{chunk_count}</div>'
                    f'<div style="font-size: 0.6rem; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;">Chunks</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
        # Footer
        st.markdown(
            '<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.1), transparent); margin: 1.5rem 0 0.75rem 0;"></div>'
            '<div style="text-align: center;">'
            '<span style="font-size: 0.62rem; color: #334155;">RAGForge v1.0 · Streamlit & Gemini</span>'
            '</div>',
            unsafe_allow_html=True
        )

def initialize_session_state():
    """Initializes common session state variables for RAGForge."""
    from utils.rag_pipeline import RAGPipeline
    
    if "rag_pipeline" not in st.session_state:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        st.session_state.rag_pipeline = RAGPipeline(api_key=api_key)
        
    if "llm_mode" not in st.session_state:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        st.session_state.llm_mode = "Gemini API" if api_key else "Mock LLM (Offline / No Key)"
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
        
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []
        
    st.session_state.ingested_docs = st.session_state.rag_pipeline.vector_store.get_document_sources()
