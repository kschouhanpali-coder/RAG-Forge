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
    page_title="RAGForge Metrics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling immediately to prevent transition blink
from utils import apply_custom_styles
apply_custom_styles()

# Heavy/slow library imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import render_sidebar, initialize_session_state, render_header

# Initialize states
initialize_session_state()

render_sidebar()
render_header()

# Page title
st.markdown("<h1>📊 RAG Pipeline Metrics & Analytics</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8; margin-bottom: 2rem;'>"
    "Audit real-time metrics of the RAG pipeline. Track retrieval vs. generation latencies, "
    "cosine similarity scores, faithfulness, recall, and overall RAG efficiency."
    "</p>",
    unsafe_allow_html=True
)

# Check if metrics are logged
if not st.session_state.metrics_history:
    st.markdown("""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 1.5rem;
        background: rgba(255, 255, 255, 0.01);
        border: 1px dashed rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.65;">📊</div>
        <span style="font-size: 1.15rem; font-weight: 800; color: #f1f5f9; display: block; margin-bottom: 0.35rem;">No Query Metrics Logged Yet</span>
        <span style="font-size: 0.85rem; color: #94a3b8; max-width: 480px; line-height: 1.6; display: block;">
            Analytics are calculated in real-time as you chat with your documents. Open the 
            <strong style="color: #a5b4fc;">Chat</strong> page, ask a few questions, and watch your metrics dashboard come alive!
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple database overview even if no queries ran yet
    st.markdown("<h3>Index Statistics</h3>", unsafe_allow_html=True)
    stats = st.session_state.rag_pipeline.vector_store.get_collection_stats()
    total_docs = stats.get("total_documents", 0)
    total_chunks = stats.get("total_chunks", 0)
    size_bytes = stats.get("collection_size_bytes", 0)
    size_kb = size_bytes / 1024.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metrics-card">'
            f'<div class="metrics-label">Total Unique Documents</div>'
            f'<div class="metrics-value">{total_docs}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metrics-card">'
            f'<div class="metrics-label">Total Indexed Chunks</div>'
            f'<div class="metrics-value">{total_chunks}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="metrics-card">'
            f'<div class="metrics-label">Collection Text Size</div>'
            f'<div class="metrics-value">{size_kb:.2f} KB</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
    st.markdown("<h3>RAG Pipeline Metrics Explained</h3>", unsafe_allow_html=True)
    
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("""
        <div class="rf-card">
            <span style="font-size: 1rem; font-weight: 700; color: #818cf8; display: flex; align-items: center; gap: 8px;">
                🔍 Retrieval Precision
            </span>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                Measures the cosine similarity of the retrieved context chunks. High similarity scores mean the vector store search is finding matches that are highly aligned with the query terms.
            </p>
        </div>
        <div class="rf-card">
            <span style="font-size: 1rem; font-weight: 700; color: #f472b6; display: flex; align-items: center; gap: 8px;">
                🧠 Answer Relevancy
            </span>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                Calculates the TF-IDF cosine similarity between the user's query and the LLM's answer. This evaluates how directly the generated response addresses the vocabulary of the question.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_desc2:
        st.markdown("""
        <div class="rf-card">
            <span style="font-size: 1rem; font-weight: 700; color: #34d399; display: flex; align-items: center; gap: 8px;">
                🛡️ Faithfulness Score
            </span>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                Audits hallucinations by computing token intersection between the LLM's response and the retrieved context. High scores verify that the model is only using provided information.
            </p>
        </div>
        <div class="rf-card">
            <span style="font-size: 1rem; font-weight: 700; color: #fb923c; display: flex; align-items: center; gap: 8px;">
                📖 Context Recall
            </span>
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                Evaluates context utilization by measuring the proportion of the retrieved context terms referenced in the answer, showing if any retrieved data was ignored.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Convert history list to pandas DataFrame
    df = pd.DataFrame(st.session_state.metrics_history)
    
    # 1. Summary Metrics Cards (Latest Query)
    latest_metrics = st.session_state.metrics_history[-1]
    
    st.markdown("<h3>Latest Query Performance</h3>", unsafe_allow_html=True)
    st.markdown(f"**Query:** `\"{latest_metrics['query']}\"`")
    
    col_metric1, col_metric2, col_metric3, col_metric4, col_metric5 = st.columns(5)
    with col_metric1:
        st.metric("Retrieval Precision", f"{latest_metrics['retrieval_precision']:.4f}")
    with col_metric2:
        st.metric("Answer Relevancy", f"{latest_metrics['answer_relevancy']:.4f}")
    with col_metric3:
        st.metric("Faithfulness Score", f"{latest_metrics['faithfulness_score']:.4f}")
    with col_metric4:
        st.metric("Context Recall", f"{latest_metrics['context_recall']:.4f}")
    with col_metric5:
        st.metric("Latency", f"{latest_metrics['total_latency']:.3f} s")
        
    # Overall Efficiency Score for the latest query
    latest_overall = latest_metrics['overall_score']
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Latest Query Overall RAG Score:** `{latest_overall:.4f}`")
    st.progress(max(0.0, min(1.0, latest_overall)), text=f"Overall Efficiency: {latest_overall * 100:.1f}%")
    
    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
    
    # 2. Charts Section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("<h3>Metric Trends Over Queries</h3>", unsafe_allow_html=True)
        # Line chart of metrics over time
        df_trends = df.copy()
        df_trends["Query Index"] = df_trends.index + 1
        
        # Melt dataframe to long-form for plotly express line
        df_melted = pd.melt(
            df_trends, 
            id_vars=["Query Index", "timestamp"], 
            value_vars=["retrieval_precision", "answer_relevancy", "faithfulness_score", "context_recall", "overall_score"],
            var_name="Metric", 
            value_name="Score"
        )
        
        # Human readable names
        metric_names = {
            "retrieval_precision": "Retrieval Precision",
            "answer_relevancy": "Answer Relevancy",
            "faithfulness_score": "Faithfulness Score",
            "context_recall": "Context Recall",
            "overall_score": "Overall RAG Score"
        }
        df_melted["Metric"] = df_melted["Metric"].map(metric_names)
        
        fig_trends = px.line(
            df_melted,
            x="Query Index",
            y="Score",
            color="Metric",
            markers=True
        )
        fig_trends.update_layout(
            xaxis_title="Query Execution Sequence",
            yaxis_title="Score (0.0 - 1.0)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", dtick=1),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", range=[0, 1.05]),
            margin=dict(t=10, l=10, r=10, b=10)
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        
    with chart_col2:
        st.markdown("<h3>RAG Pipeline Strengths / Weaknesses</h3>", unsafe_allow_html=True)
        # Radar Chart of Averages
        avg_precision = df["retrieval_precision"].mean()
        avg_relevancy = df["answer_relevancy"].mean()
        avg_faithfulness = df["faithfulness_score"].mean()
        avg_recall = df["context_recall"].mean()
        avg_overall = df["overall_score"].mean()
        
        categories = ['Retrieval Precision', 'Answer Relevancy', 'Faithfulness Score', 'Context Recall']
        r_values = [avg_precision, avg_relevancy, avg_faithfulness, avg_recall, avg_precision]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_values,
            theta=categories + [categories[0]],
            fill='toself',
            name='Average Strengths',
            line_color='#a855f7',
            fillcolor='rgba(168, 85, 247, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(255,255,255,0.05)',
                    linecolor='rgba(255,255,255,0.1)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.05)',
                    linecolor='rgba(255,255,255,0.1)'
                )
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            margin=dict(t=30, l=45, r=45, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
    
    # 3. Metrics History Table & CSV Export
    st.markdown("<h3>Full Execution Metrics History</h3>", unsafe_allow_html=True)
    
    styled_df = df[[
        "timestamp", 
        "query", 
        "retrieval_precision", 
        "answer_relevancy", 
        "faithfulness_score", 
        "context_recall", 
        "overall_score",
        "total_latency"
    ]].copy()
    
    styled_df.columns = [
        "Time", 
        "User Query", 
        "Retrieval Precision", 
        "Answer Relevancy", 
        "Faithfulness Score", 
        "Context Recall", 
        "Overall Score",
        "Latency (s)"
    ]
    
    st.dataframe(
        styled_df.style.format({
            "Retrieval Precision": "{:.4f}",
            "Answer Relevancy": "{:.4f}",
            "Faithfulness Score": "{:.4f}",
            "Context Recall": "{:.4f}",
            "Overall Score": "{:.4f}",
            "Latency (s)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Download Button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Metrics History as CSV",
        data=csv_data,
        file_name="ragforge_metrics_history.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
    
    # 4. Metrics Interpretation & Optimization Suggestions
    st.markdown("<h3>Metrics Interpretation Guide</h3>", unsafe_allow_html=True)
    
    col_guide1, col_guide2 = st.columns(2)
    
    with col_guide1:
        st.markdown("""
        <div class="rf-card" style="border-left: 5px solid #22c55e;">
            <span style="font-size: 1.1rem; font-weight: 700; color: #22c55e;">🟢 Score > 0.8: Excellent RAG Pipeline</span>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                The pipeline operates at peak efficiency. Chunks are highly relevant, the LLM is answering accurately,
                no hallucinations are detected (high faithfulness), and the context is fully utilized.
            </p>
        </div>
        <div class="rf-card" style="border-left: 5px solid #eab308;">
            <span style="font-size: 1.1rem; font-weight: 700; color: #eab308;">🟡 Score 0.5 - 0.8: Good RAG Pipeline</span>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                The pipeline is functional but has room for tuning. Usually, tweaking parameters like Top-K chunks,
                reducing noise in text parsing, or refining the prompts will push scores into the excellent range.
            </p>
        </div>
        <div class="rf-card" style="border-left: 5px solid #ef4444;">
            <span style="font-size: 1.1rem; font-weight: 700; color: #ef4444;">🔴 Score < 0.5: Needs Improvement</span>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; line-height: 1.5;">
                The pipeline has high retrieval mismatch, hallucinations, or irrelevant answers. Review
                the optimization recommendations to target weak pipeline areas.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_guide2:
        st.markdown("<h4>🛠️ Targeted RAG Optimization Suggestions</h4>", unsafe_allow_html=True)
        
        # Calculate weakest metric
        avgs = {
            "Retrieval Precision": avg_precision,
            "Answer Relevancy": avg_relevancy,
            "Faithfulness Score": avg_faithfulness,
            "Context Recall": avg_recall
        }
        weakest_metric = min(avgs, key=avgs.get)
        weakest_val = avgs[weakest_metric]
        
        st.info(f"**Analysis:** Your current weakest pipeline metric is **{weakest_metric}** (Average Score: `{weakest_val:.4f}`).")
        
        if weakest_metric == "Retrieval Precision":
            st.markdown("""
            **How to improve Retrieval Precision:**
            1. **Re-evaluate Chunk Sizes**: If chunk size is too small, similarity matches can get noisy. Consider increasing chunk size.
            2. **Increase Chunk Overlap**: Setting higher overlap (e.g. 75-100 characters) keeps contextual bounds from splitting across chunk borders.
            3. **Clean Noise**: Clean files before indexing by stripping extraneous formatting or code tags that inflate TF-IDF terms.
            """)
        elif weakest_metric == "Answer Relevancy":
            st.markdown("""
            **How to improve Answer Relevancy:**
            1. **Refine Prompt Templates**: Check your system instruction template. Add strict phrasing to force the LLM to direct its vocabulary specifically toward addressing the user's explicit question.
            2. **Filter Outliers**: Filter retrieved chunks with a similarity threshold to keep low-scoring irrelevant text from being fed as noise.
            """)
        elif weakest_metric == "Faithfulness Score":
            st.markdown("""
            **How to improve Faithfulness (Reduce Hallucinations):**
            1. **Lower LLM Temperature**: Drop LLM temperature to `0.0` or `0.1` in the Chat page. This makes Gemini deterministic and strictly grounded.
            2. **Add Strict Constraints**: Update the prompt instruction to include: *"Answer the question using ONLY the provided context. If the answer cannot be found in the context, state 'I don't know'."*
            """)
        elif weakest_metric == "Context Recall":
            st.markdown("""
            **How to improve Context Recall:**
            1. **Increase Top-K Chunks**: Instruct the system to fetch more chunks (e.g. top 5-6 instead of top 3) to supply comprehensive source coverage.
            2. **Exhaustive Prompt Instruction**: instruct the LLM: *"Ensure you cite and utilize key facts, figures, and definitions from all sources provided in the context."*
            """)
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Clear stats
        if st.button("🧹 Clear Performance History"):
            st.session_state.metrics_history = []
            st.success("Successfully reset RAG metrics database!")
            st.rerun()
