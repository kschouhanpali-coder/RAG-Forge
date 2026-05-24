# 🔮 RAGForge — Pipeline Laboratory

> An interactive studio for designing, debugging, and analyzing **Retrieval-Augmented Generation (RAG)** workflows. Understand exactly how text preprocessing, vector indexing, TF-IDF similarity search, and LLM generation work in tandem.

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://rag-forge-au3fkjxa6oqcfyvk246t7n.streamlit.app/Chat)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-00C9A7?style=for-the-badge)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

</div>

---

## 🌐 Live Demo

**👉 [https://rag-forge-au3fkjxa6oqcfyvk246t7n.streamlit.app/Chat](https://rag-forge-au3fkjxa6oqcfyvk246t7n.streamlit.app/Chat)**

Try the full pipeline live — no setup required. Upload a document, ask questions, and watch every step of the RAG pipeline in real time.

---

## 📸 Screenshots

| Dashboard | Chat Laboratory |
|-----------|----------------|
| Pipeline architecture overview with 6-stage flow | Real-time Q&A with chunk trace and similarity scores |

| Document Ingestion | RAG Metrics |
|-------------------|-------------|
| Upload PDFs, DOCX, TXT with live chunking progress | Per-query precision, faithfulness, recall & latency charts |

---

## ✨ Features

### 💬 Chat Laboratory
- Query your knowledge base in real time
- Inspect the **exact chunks** that informed each answer with cosine similarity scores
- Adjust **Top-K chunks** and **LLM temperature** live via sliders
- Full **Pipeline Trace** — preprocessed query, TF-IDF dimensions, retrieval & generation latency

### 📄 Document Ingestion
- Upload **PDF, DOCX, TXT** files
- Watch text get extracted, chunked, and vectorized step by step
- Verify ingestion with a built-in similarity search test
- Persistent storage via **ChromaDB**

### 🔗 Web Scraping & URL Ingestion
- Paste any URL — RAGForge strips scripts, ads, and navigation noise
- Supports **batch URL ingestion** (one per line)
- Auto RAG verification after every ingest job
- Scraped text preview with indexed chunk details

### 📊 RAG Pipeline Metrics & Analytics
- **Per-query metrics:** Retrieval Precision · Answer Relevancy · Faithfulness Score · Context Recall · Latency
- **Trend line chart** across query execution sequence
- **Radar spider chart** for pipeline strength/weakness visualization
- Full execution history table with **CSV export**
- Targeted optimization suggestions based on your weakest metric

---

## 🏗️ Pipeline Architecture

```
User Query
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐
│  Query  │───▶│  Clean  │───▶│ TF-IDF  │───▶│ Retrieve │───▶│  Rank  │───▶│ Generate │
│  Input  │    │Preprocess│   │Vectorize│    │ ChromaDB │    │Cos.Sim │    │  Gemini  │
└─────────┘    └─────────┘    └─────────┘    └──────────┘    └────────┘    └──────────┘
```

| Stage | Description |
|-------|-------------|
| **Query** | Raw user input received |
| **Clean** | Lowercased, stop words removed, noise stripped |
| **TF-IDF** | Converted to dense float vectors via scikit-learn |
| **Retrieve** | ChromaDB cosine similarity search returns top-K chunks |
| **Rank** | Chunks scored and ranked by relevance |
| **Generate** | Google Gemini 1.5 Flash generates a grounded answer |

---

## 📐 Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| **Retrieval Precision** | How relevant retrieved chunks are to the query | > 0.8 |
| **Answer Relevancy** | How well the answer addresses the question | > 0.8 |
| **Faithfulness Score** | Whether the answer is grounded in the retrieved context | > 0.8 |
| **Context Recall** | Proportion of relevant chunks successfully retrieved | > 0.8 |
| **Overall RAG Score** | Weighted average of all above metrics | > 0.8 |

### Score Interpretation
- 🟢 **> 0.8** — Excellent: pipeline operating at peak efficiency
- 🟡 **0.5 – 0.8** — Good: functional but room for tuning (adjust Top-K, chunk overlap)
- 🔴 **< 0.5** — Needs Improvement: review retrieval parameters and document quality

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ragforge.git
cd ragforge

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Configuration

1. Launch the app
2. Enter your **Gemini API Key** in the sidebar
3. Confirm **API Connected** status turns green
4. Start ingesting documents or URLs!

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Google Gemini 1.5 Flash |
| **Vector Store** | ChromaDB |
| **Vectorizer** | scikit-learn TF-IDF |
| **Document Parsing** | python-docx, PyPDF2, BeautifulSoup4 |
| **Metrics** | Custom RAG evaluation pipeline |

---

## 💡 Optimization Tips

If your **Retrieval Precision** is low (< 0.5):
1. **Re-evaluate Chunk Sizes** — increase chunk size if similarity matches are noisy
2. **Increase Chunk Overlap** — set 75–100 character overlap to preserve context across chunk borders
3. **Clean Noise** — strip extraneous formatting or code tags before indexing to avoid inflating TF-IDF terms

---

## 📁 Project Structure

```
ragforge/
├── app.py                  # Main Streamlit entry point
├── pages/
│   ├── Chat.py             # Chat Laboratory
│   ├── Document_Upload.py  # Document Ingestion
│   ├── URL_Ingestion.py    # Web Scraping
│   └── RAG_Metrics.py      # Analytics Dashboard
├── utils/
│   ├── vectorizer.py       # TF-IDF vectorization
│   ├── retriever.py        # ChromaDB retrieval
│   ├── generator.py        # Gemini LLM integration
│   └── metrics.py          # RAG evaluation metrics
├── requirements.txt
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini/) for the LLM backbone
- [ChromaDB](https://www.trychroma.com/) for persistent vector storage
- [Streamlit](https://streamlit.io/) for the interactive UI framework
- [scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization

---

<div align="center">

**RAGForge v1.0.0** · Built with ❤️ using Streamlit & Gemini

[![Live Demo](https://img.shields.io/badge/Try_it_Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rag-forge-au3fkjxa6oqcfyvk246t7n.streamlit.app/Chat)

</div>
