import os
import sys

# Configure streamlit to use local workspace directory for configuration and credentials
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
    page_title="RAGForge URL Ingestion",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling immediately to prevent transition blink
from utils import apply_custom_styles
apply_custom_styles()

# Heavy/slow library imports
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from utils import render_sidebar, initialize_session_state, render_header

# Initialize states
initialize_session_state()

render_sidebar()
render_header()

# Page title
st.markdown("<h1>🔗 Web Scraping & URL Ingestion</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8; margin-bottom: 2rem;'>"
    "Ingest multiple websites simultaneously. Enter URLs one per line, extract clean article prose, "
    "and verify matching search responses automatically."
    "</p>",
    unsafe_allow_html=True
)

# Helper function to scrape and parse web HTML
def scrape_url_content(html_content: str) -> tuple[str, str]:
    """
    Parses HTML content, strips headers/footers/scripts, and extracts text
    specifically from article, p, h1, h2, and h3 tags.
    Filters out common script/javascript disabled fallback warnings.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Get page title
    page_title = soup.title.string if soup.title else "Untitled Webpage"
    
    # Remove script, style, nav, footer, header tags
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
        
    # Common JavaScript/noscript fallback or browser/cookie warning phrases to filter out
    js_warning_phrases = [
        "javascript did not run",
        "interactive scripts did not run",
        "disabled javascript",
        "enable javascript",
        "javascript is disabled",
        "turn on javascript",
        "browser does not support javascript",
        "unsupported browser",
        "activate javascript",
        "please enable js",
        "cookies are disabled",
        "enable cookies"
    ]
    
    # Extract clean text from specific tags
    content_tags = soup.find_all(["p", "h1", "h2", "h3", "article"])
    text_parts = []
    for tag in content_tags:
        text = tag.get_text().strip()
        if not text:
            continue
        text_lower = text.lower()
        # Skip this element if it matches any javascript warning/fallback phrases
        if any(phrase in text_lower for phrase in js_warning_phrases):
            continue
        text_parts.append(text)
        
    # Normalize whitespaces
    full_text = "\n".join(text_parts)
    full_text = re.sub(r'\s+', ' ', full_text)
    return page_title.strip(), full_text.strip()

# Layout: Ingest Control & Dynamic Live Progress Table
st.markdown("<h3>Scrape Web Pages</h3>", unsafe_allow_html=True)

urls_input = st.text_area(
    "Enter Website URLs (One URL per line)",
    height=150,
    placeholder="https://example.com/blog/article1\nhttps://example.com/blog/article2",
    help="Each URL should be fully qualified starting with http:// or https://"
)

ingest_button = st.button("🌐 Ingest & Index URLs", use_container_width=True)

if ingest_button and urls_input:
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
    
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        # Create placeholders for live status reporting
        status_table_placeholder = st.empty()
        detailed_expanders_placeholder = st.container()
        
        # Ingestion loop metrics
        scraping_results = []
        detailed_logs = {} # Store info to display in expanders later
        
        for idx, url in enumerate(urls):
            start_time = time.time()
            status = "⏳ Ingesting..."
            num_chunks = 0
            test_query_result = "N/A"
            
            # Initial placeholder update
            current_results = scraping_results + [{
                "URL": url,
                "Status": status,
                "Chunks": num_chunks,
                "Time": "0.00s",
                "Test Query Result": test_query_result
            }]
            status_table_placeholder.dataframe(pd.DataFrame(current_results), use_container_width=True, hide_index=True)
            
            # 1. Invalid URL Check
            if not (url.startswith("http://") or url.startswith("https://")):
                status = "Invalid URL ❌"
                elapsed = time.time() - start_time
                scraping_results.append({
                    "URL": url,
                    "Status": status,
                    "Chunks": 0,
                    "Time": f"{elapsed:.2f}s",
                    "Test Query Result": "N/A"
                })
                continue
                
            try:
                # Fetch Webpage
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                # 2. Status check
                if response.status_code == 403:
                    status = "Blocked (403) ❌"
                    elapsed = time.time() - start_time
                    scraping_results.append({
                        "URL": url,
                        "Status": status,
                        "Chunks": 0,
                        "Time": f"{elapsed:.2f}s",
                        "Test Query Result": "N/A"
                    })
                    continue
                elif response.status_code != 200:
                    status = f"Error ({response.status_code}) ❌"
                    elapsed = time.time() - start_time
                    scraping_results.append({
                        "URL": url,
                        "Status": status,
                        "Chunks": 0,
                        "Time": f"{elapsed:.2f}s",
                        "Test Query Result": "N/A"
                    })
                    continue
                    
                # 3. Parse HTML
                page_title, clean_text = scrape_url_content(response.text)
                
                if not clean_text:
                    status = "Empty Content ⚠️"
                    elapsed = time.time() - start_time
                    scraping_results.append({
                        "URL": url,
                        "Status": status,
                        "Chunks": 0,
                        "Time": f"{elapsed:.2f}s",
                        "Test Query Result": "N/A"
                    })
                    continue
                    
                # 4. RAG Index Ingestion
                vector_store = st.session_state.rag_pipeline.vector_store
                new_chunks = vector_store.chunk_text(clean_text)
                
                if not new_chunks:
                    status = "Empty Chunks ⚠️"
                    elapsed = time.time() - start_time
                    scraping_results.append({
                        "URL": url,
                        "Status": status,
                        "Chunks": 0,
                        "Time": f"{elapsed:.2f}s",
                        "Test Query Result": "N/A"
                    })
                    continue
                    
                # Merge new chunks with existing database corpus
                existing_data = vector_store.collection.get(include=["documents", "metadatas"])
                existing_docs = existing_data.get("documents", [])
                existing_metas = existing_data.get("metadatas", [])
                
                all_chunks = list(existing_docs) + new_chunks
                
                # Build metadatas
                all_metas = list(existing_metas)
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                for c_i, chunk in enumerate(new_chunks):
                    all_metas.append({
                        "source": url,
                        "chunk_index": c_i,
                        "timestamp": current_time
                    })
                    
                # Vectorize & Store (re-fits and updates vocabulary in vectorizer)
                vectors = vector_store.vectorize(all_chunks)
                vector_store.store(all_chunks, vectors, all_metas)
                
                # Update status
                status = "Success ✅"
                num_chunks = len(new_chunks)
                elapsed = time.time() - start_time
                
                # 5. Auto-run test RAG query on scraped content
                test_query = f"What is the main summary of {page_title}?"
                if not st.session_state.rag_pipeline.llm.is_configured():
                    test_answer = "⚠️ Ingestion success. Verification skipped (Missing API key)."
                    test_query_result = "Skipped (API Key)"
                    test_sources = []
                else:
                    # Run RAG search + Gemini generation
                    test_answer, test_sources, test_scores, test_metrics = st.session_state.rag_pipeline.query(test_query, top_k=2)
                    test_query_result = test_answer[:80] + "..." if len(test_answer) > 80 else test_answer
                
                scraping_results.append({
                    "URL": url,
                    "Status": status,
                    "Chunks": num_chunks,
                    "Time": f"{elapsed:.2f}s",
                    "Test Query Result": test_query_result
                })
                
                # Store logs for the expander rendering
                detailed_logs[url] = {
                    "title": page_title,
                    "text_preview": clean_text[:500] + ("..." if len(clean_text) > 500 else ""),
                    "chunks_count": num_chunks,
                    "test_query": test_query,
                    "test_answer": test_answer if st.session_state.rag_pipeline.llm.is_configured() else "Missing Gemini API Key.",
                    "test_sources": test_sources
                }
                
            except requests.exceptions.Timeout:
                status = "Timeout ❌"
                elapsed = time.time() - start_time
                scraping_results.append({
                    "URL": url,
                    "Status": status,
                    "Chunks": 0,
                    "Time": f"{elapsed:.2f}s",
                    "Test Query Result": "N/A"
                })
            except Exception as e:
                status = f"Failed ({type(e).__name__}) ❌"
                elapsed = time.time() - start_time
                scraping_results.append({
                    "URL": url,
                    "Status": status,
                    "Chunks": 0,
                    "Time": f"{elapsed:.2f}s",
                    "Test Query Result": "N/A"
                })
                
            # Keep updating the dataframe placeholder in real-time
            status_table_placeholder.dataframe(pd.DataFrame(scraping_results), use_container_width=True, hide_index=True)
            
        # Synchronize session state sources list
        st.session_state.ingested_docs = st.session_state.rag_pipeline.vector_store.get_document_sources()
        
        # Display details and expanders after scraping finishes
        with detailed_expanders_placeholder:
            st.markdown("<h3>Scraped URL Details</h3>", unsafe_allow_html=True)
            for url, log_data in detailed_logs.items():
                with st.expander(f"🌐 Detail: {log_data['title']} ({url})", expanded=False):
                    st.markdown(f"**Webpage Title:** `{log_data['title']}`")
                    st.markdown(f"**Indexed Chunks:** `{log_data['chunks_count']}` chunks created")
                    st.markdown("**Scraped Text Preview:**")
                    import html
                    escaped_preview = html.escape(log_data["text_preview"])
                    st.markdown(f'<div class="rf-chunk-text">{escaped_preview}</div>', unsafe_allow_html=True)
                    
                    # Ingestion Verification Trace Panel
                    st.markdown("#### 🔄 Ingestion RAG Verification Trace")
                    st.markdown(f"**Test Verification Query:** `{log_data['test_query']}`")
                    st.markdown(f"**Gemini Reasoning Output:**\n>{log_data['test_answer']}")
                    
                    if log_data["test_sources"]:
                        st.markdown("**Retrieved Context Verification Chunks:**")
                        import html
                        for s_i, chunk_src in enumerate(log_data["test_sources"]):
                            src_name = chunk_src["metadata"].get("source", "Unknown")
                            similarity_score = chunk_src.get("similarity_score", 0.0)
                            pct = max(0.0, min(1.0, similarity_score)) * 100
                            escaped_content = html.escape(chunk_src["page_content"])
                            
                            card_html = f"""
                            <div class="rf-match-card">
                                <div class="rf-match-header">
                                    <span class="rf-match-badge">Match {s_i+1}</span>
                                    <span class="rf-match-source" title="{src_name}">{src_name}</span>
                                    <span class="rf-match-score">Score: {similarity_score:.4f}</span>
                                </div>
                                <div class="rf-match-bar-container">
                                    <div class="rf-match-bar-fill" style="width: {pct}%;"></div>
                                </div>
                                <div class="rf-match-body">{escaped_content}</div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                            
        # Final knowledge base stats
        st.success("🎉 URL Ingestion Job completed!")
        st.markdown("### Total Knowledge Base Stats")
        final_stats = st.session_state.rag_pipeline.vector_store.get_collection_stats()
        final_size_kb = final_stats.get("collection_size_bytes", 0) / 1024.0
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Indexed Documents", final_stats.get("total_documents", 0))
        with col_s2:
            st.metric("Total Indexed Chunks", final_stats.get("total_chunks", 0))
        with col_s3:
            st.metric("Index Text Size", f"{final_size_kb:.2f} KB")

# Display currently indexed web sources
st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
st.markdown("<h3>Currently Indexed URL Sources</h3>", unsafe_allow_html=True)
indexed_sources = st.session_state.rag_pipeline.vector_store.get_document_sources()
url_sources = [s for s in indexed_sources if s.startswith("http://") or s.startswith("https://")]

if not url_sources:
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
        margin: 1rem 0;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.6;">🌐</div>
        <span style="font-size: 0.95rem; font-weight: 700; color: #cbd5e1; display: block; margin-bottom: 0.25rem;">No Web Sources Scraped</span>
        <span style="font-size: 0.78rem; color: #64748b; max-width: 280px; line-height: 1.5; display: block;">
            Enter webpage URLs above to scrape and index content into your vector store.
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    for source in url_sources:
        col_url_name, col_url_del = st.columns([5, 1], vertical_alignment="center")
        with col_url_name:
            st.markdown(
                f'<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 0.6rem 0.8rem; border-radius: 8px; font-size: 0.88rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">'
                f'🔗 <a href="{source}" target="_blank" style="color: #818cf8; text-decoration: none;">{source}</a>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_url_del:
            if st.button("🗑️", key=f"del_url_{source}", help=f"Delete URL: {source}", use_container_width=True):
                st.session_state.rag_pipeline.delete_document(source)
                st.success(f"Removed `{source}`")
                st.session_state.ingested_docs = st.session_state.rag_pipeline.vector_store.get_document_sources()
                st.rerun()
