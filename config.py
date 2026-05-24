import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App Directories & Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = "ragforge_collection"

# LLM Configuration
GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 4
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# RAG Template
PROMPT_TEMPLATE = """You are a professional assistant. Use the following context to answer the question. Your answer must be written in a single, well-structured paragraph. Do not use lists, bullet points, or multiple paragraphs. If the context does not contain the specific details to answer the question, state clearly that the retrieved documents do not contain information to answer the question, but mention what they do contain if relevant. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

# UI Styles (Custom CSS for Premium Dark Theme)
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #0a0c15;
    --bg-secondary: #0f1221;
    --bg-card: rgba(15, 18, 33, 0.65);
    --bg-card-hover: rgba(20, 24, 45, 0.85);
    --border-subtle: rgba(99, 102, 241, 0.08);
    --border-accent: rgba(99, 102, 241, 0.25);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-pink: #ec4899;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    --gradient-warm: linear-gradient(135deg, #f472b6 0%, #ec4899 50%, #db2777 100%);
    --gradient-cool: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease, transform 0.2s ease, color 0.2s ease, opacity 0.2s ease;
}

/* ===== MAIN APP ===== */
.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 50%, rgba(139, 92, 246, 0.04), transparent);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080a12 0%, #0c0e18 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: unset !important;
    background: none !important;
}

/* ===== TYPOGRAPHY ===== */
h1 {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #c084fc 60%, #f472b6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    font-size: 2.2rem !important;
    line-height: 1.2 !important;
    padding-bottom: 0.25rem;
}

h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 50%, #e879f9 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

p, li, span, div {
    font-family: 'Inter', sans-serif;
}

/* ===== INPUT FIELDS ===== */
div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background: rgba(15, 18, 33, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    backdrop-filter: blur(8px);
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 1px var(--accent-indigo), var(--shadow-glow) !important;
}

/* ===== BUTTONS ===== */
button[kind="secondary"], button[kind="primary"],
.stButton > button {
    background: var(--gradient-primary) !important;
    border: none !important;
    color: white !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
    position: relative;
    overflow: hidden;
}

button[kind="secondary"]:hover, button[kind="primary"]:hover,
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(1.1);
}

button[kind="secondary"]:active, button[kind="primary"]:active,
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ===== PREMIUM CARD ===== */
.rf-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    position: relative;
    overflow: hidden;
}

.rf-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
}

.rf-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-accent);
    box-shadow: var(--shadow-glow);
    background: var(--bg-card-hover);
}

/* ===== BLOCK BORDER CONTAINER OVERRIDE ===== */
div[data-testid="stBlockBorderContainer"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stBlockBorderContainer"]:hover {
    border-color: var(--border-accent) !important;
    box-shadow: var(--shadow-glow) !important;
}


/* ===== CHAT BUBBLES ===== */
div[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}

div[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.15) !important;
}

/* ===== METRICS CARDS ===== */
.metrics-card {
    background: linear-gradient(145deg, rgba(15, 18, 33, 0.8) 0%, rgba(25, 20, 55, 0.8) 100%);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.metrics-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metrics-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
}

.metrics-card:hover::after {
    opacity: 1;
}

.metrics-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f472b6 0%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.25rem;
    line-height: 1.2;
}

.metrics-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ===== ST.METRIC OVERRIDE ===== */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: var(--border-accent) !important;
    box-shadow: var(--shadow-glow);
}

div[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* ===== EXPANDER ===== */
details[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s ease;
}

details[data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.15) !important;
}

details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
}

/* ===== DATAFRAME TABLE ===== */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ===== FILE UPLOADER ===== */
div[data-testid="stFileUploader"] {
    border-radius: var(--radius-md) !important;
}

div[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(99, 102, 241, 0.2) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(15, 18, 33, 0.5) !important;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    background: rgba(20, 24, 45, 0.6) !important;
}

/* ===== PROGRESS BAR ===== */
div[data-testid="stProgress"] > div > div > div > div {
    background: var(--gradient-primary) !important;
    border-radius: 99px !important;
}
div[data-testid="stProgress"] > div > div > div {
    background-color: rgba(99, 102, 241, 0.1) !important;
    border-radius: 99px !important;
}

/* ===== STATUS & ALERTS ===== */
div[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border: none !important;
    font-size: 0.9rem;
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] {
    border-radius: var(--radius-sm) !important;
}

/* ===== CHAT INPUT ===== */
div[data-testid="stChatInput"] {
    border: 1px solid rgba(99, 102, 241, 0.18) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(15, 18, 33, 0.85) !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 1px var(--accent-indigo), 0 -4px 24px rgba(99, 102, 241, 0.15) !important;
}

/* Remove duplicate inner borders inside chat input */
div[data-testid="stChatInput"] div[data-baseweb="textarea"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ===== SLIDER ===== */
div[data-testid="stSlider"] > div > div > div {
    color: var(--accent-violet) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.2);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.4);
}

/* ===== CODE BLOCKS ===== */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

pre {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ===== FOOTER ===== */
.footer-text {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 2rem;
    border-top: 1px solid var(--border-subtle);
    padding-top: 1rem;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(99, 102, 241, 0.1); }
    50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.25); }
}

.animate-in {
    animation: fadeInUp 0.5s ease-out;
}

/* ===== TAB STYLING ===== */
button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
}

.stDownloadButton > button:hover {
    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.4) !important;
}

.rf-chunk-text {
    background: rgba(15, 18, 33, 0.45) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 8px !important;
    padding: 0.85rem 1.1rem !important;
    font-size: 0.88rem !important;
    color: #cbd5e1 !important;
    line-height: 1.6 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    font-family: 'Inter', sans-serif !important;
}

/* ===== DESTRUCTIVE & DELETE BUTTONS ===== */
.st-key-clear_db_btn button, .st-key-btn_yes button, .st-key-del_doc_btn button, div[class*="st-key-del_"] button {
    background: rgba(239, 68, 68, 0.02) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    color: #f87171 !important;
    box-shadow: none !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease !important;
}

.st-key-clear_db_btn button:hover, .st-key-btn_yes button:hover, .st-key-del_doc_btn button:hover, div[class*="st-key-del_"] button:hover {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15) !important;
}

/* Compact layout for inline trash buttons */
div[class*="st-key-del_"] button {
    padding: 0.4rem 0.5rem !important;
    height: 38px !important;
    line-height: 1 !important;
    border-radius: 8px !important;
}

/* ===== CUSTOM SIDEBAR PAGE LINKS ===== */
[data-testid="stPageLink-NavLink"] {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.85rem !important;
    margin-bottom: 0.35rem !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease, transform 0.2s ease !important;
}

[data-testid="stPageLink-NavLink"]:hover {
    background: rgba(99, 102, 241, 0.06) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
    color: #a5b4fc !important;
    transform: translateX(4px) !important;
}

/* Active state link highlights */
[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    color: #cbd5e1 !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.08) !important;
}

/* ===== PIPELINE FLOW CHART ===== */
.rf-pipeline-container {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 1.25rem 1.5rem;
    background: rgba(15, 18, 33, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 14px;
    overflow-x: auto;
    scrollbar-width: none; /* Firefox */
}

.rf-pipeline-container::-webkit-scrollbar {
    display: none; /* Safari/Chrome */
}

.rf-pipeline-node {
    flex: 1 1 0%;
    min-width: 90px;
    max-width: 130px;
    background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(139,92,246,0.04) 100%);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 0.85rem 0.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.rf-pipeline-node:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.15);
}

.rf-pipeline-node.gemini {
    background: linear-gradient(135deg, rgba(236,72,153,0.08) 0%, rgba(168,85,247,0.05) 100%);
    border-color: rgba(236, 72, 153, 0.15);
}

.rf-pipeline-node.gemini:hover {
    border-color: rgba(236, 72, 153, 0.4) !important;
    box-shadow: 0 6px 16px rgba(236, 72, 153, 0.15);
}

.rf-pipeline-arrow {
    color: rgba(99, 102, 241, 0.3);
    font-size: 1.1rem;
    font-weight: bold;
    flex-shrink: 0;
    user-select: none;
    animation: rfArrowPulse 2s infinite ease-in-out;
}

@keyframes rfArrowPulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50% { opacity: 0.9; transform: scale(1.15); }
}

/* ===== RAG MATCH CARDS ===== */
.rf-match-card {
    background: rgba(15, 18, 33, 0.45) !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
    margin-top: 0.65rem !important;
    margin-bottom: 0.65rem !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.rf-match-card:hover {
    border-color: rgba(99, 102, 241, 0.28) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.12) !important;
    transform: translateY(-2px) !important;
}
.rf-match-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.65rem;
    flex-wrap: wrap;
    gap: 8px;
}
.rf-match-badge {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.rf-match-source {
    color: #a5b4fc;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(99, 102, 241, 0.05);
    padding: 0.15rem 0.5rem;
    border-radius: 5px;
    border: 1px solid rgba(99, 102, 241, 0.1);
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.rf-match-score {
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
    background: rgba(52, 211, 153, 0.08);
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    border: 1px solid rgba(52, 211, 153, 0.15);
    font-family: 'JetBrains Mono', monospace;
}
.rf-match-bar-container {
    width: 100%;
    height: 5px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 3px;
    margin-bottom: 0.75rem;
    overflow: hidden;
}
.rf-match-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    border-radius: 3px;
}
.rf-match-body {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'Inter', sans-serif !important;
}
</style>
"""

# Logo and Visual Elements
LOGO_HTML = """
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem;">
    <svg width="42" height="42" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0; filter: drop-shadow(0px 4px 12px rgba(99, 102, 241, 0.45));">
        <defs>
            <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#6366f1" />
                <stop offset="50%" stop-color="#8b5cf6" />
                <stop offset="100%" stop-color="#ec4899" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <rect x="2" y="2" width="96" height="96" rx="22" fill="#0b0d19" stroke="url(#logoGrad)" stroke-width="3"/>
        <circle cx="50" cy="50" r="32" stroke="url(#logoGrad)" stroke-width="2" stroke-dasharray="6 8" />
        <line x1="32" y1="50" x2="52" y2="34" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
        <line x1="32" y1="50" x2="52" y2="66" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
        <line x1="52" y1="34" x2="70" y2="50" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
        <line x1="52" y1="66" x2="70" y2="50" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"/>
        <line x1="52" y1="34" x2="52" y2="66" stroke="url(#logoGrad)" stroke-width="1.5" stroke-dasharray="3 3"/>
        <circle cx="32" cy="50" r="5" fill="#f1f5f9"/>
        <circle cx="52" cy="34" r="5" fill="#818cf8" filter="url(#glow)"/>
        <circle cx="52" cy="66" r="5" fill="#f472b6" filter="url(#glow)"/>
        <circle cx="70" cy="50" r="6" fill="url(#logoGrad)" filter="url(#glow)"/>
    </svg>
    <div>
        <span style="
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem; font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
        ">RAGForge</span>
        <div style="
            font-size: 0.65rem; color: #8b5cf6;
            font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; margin-top: -3px;
        ">Pipeline Laboratory</div>
    </div>
</div>
"""

# Clean up CUSTOM_CSS to remove empty/blank lines to prevent Streamlit markdown parser from outputting raw CSS text
CUSTOM_CSS = "\n".join([line for line in CUSTOM_CSS.split("\n") if line.strip()])
