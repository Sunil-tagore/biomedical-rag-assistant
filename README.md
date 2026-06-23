# 🧬 Biomedical Research Paper RAG Assistant

A domain-specific Retrieval-Augmented Generation (RAG) system that enables researchers to upload biomedical research papers and ask natural language questions — receiving source-grounded AI answers powered by Llama 3.3 70B.

**🔗 Live Demo:** [huggingface.co/spaces/sunilveluturi/biomedical-rag-assistant](https://huggingface.co/spaces/sunilveluturi/biomedical-rag-assistant)

---

## 🎯 What It Does

Upload any biomedical research paper PDF and ask questions like:
- *"What trajectory inference methods were benchmarked?"*
- *"What are the four evaluation criteria used?"*
- *"Which methods performed best for scalability?"*

The system retrieves the most relevant sections from the paper and generates a precise, source-grounded answer — without hallucinating information not present in the document.

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Smart Cleaning + Chunking (LangChain RecursiveCharacterTextSplitter)
    ↓
Embeddings (Hugging Face sentence-transformers/all-MiniLM-L6-v2)
    ↓
Vector Store (ChromaDB)
    ↓
Hybrid Retrieval (Semantic Search + Keyword Search)
    ↓
Answer Generation (Llama 3.3 70B via Groq Inference API)
    ↓
Source-Grounded Response + Evidence Chunks
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Hugging Face sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Llama 3.3 70B via Groq Inference API |
| Retrieval | Hybrid — Semantic Vector Search + Keyword Search |
| Frontend | Streamlit |
| Deployment | Docker + Hugging Face Spaces |
| PDF Processing | PyPDF |

---

## ✨ Key Features

- **Hybrid Retrieval** — combines semantic vector search and keyword-based search for higher recall on domain-specific biomedical terminology
- **Noise Filtering** — automatically removes references, acknowledgements, and administrative sections before indexing
- **Duplicate Removal** — deduplicates retrieved chunks using signature-based matching
- **Source Evidence** — every AI answer is accompanied by the exact retrieved chunks it was generated from
- **Hallucination Reduction** — LLM is instructed to answer only from provided context, not from training data

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/Sunil-tagore/biomedical-rag-assistant.git
cd biomedical-rag-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📁 Project Structure

```
biomedical-rag-assistant/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration for HF Spaces
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── .gitignore
└── README.md
```

---

## 🧠 How RAG Works Here

1. **Ingestion** — PDF text is extracted, cleaned, and split into 1000-character chunks with 200-character overlap
2. **Embedding** — each chunk is converted into a 384-dimensional vector using MiniLM sentence-transformers
3. **Retrieval** — user question is embedded and compared against all chunk vectors using cosine similarity; top matches from both vector search and keyword search are combined
4. **Generation** — retrieved chunks are passed as context to Llama 3.3 70B with a strict prompt instructing it to answer only from the provided context

---

## 👨‍💻 Author

**Sunil Tagore Veluturi**
MS Engineering Data Science, University of Houston (GPA: 4.0)
Machine Learning Research Intern, Baylor College of Medicine

[LinkedIn](https://linkedin.com/in/sunil-tagore-veluturi) | [Hugging Face](https://huggingface.co/sunilveluturi)
