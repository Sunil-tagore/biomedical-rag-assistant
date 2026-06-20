import uuid
import html
import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Load environment variables from .env file ──
load_dotenv()

# ── Page config ──
st.set_page_config(
    page_title="Biomedical Research Paper RAG Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Biomedical Research Paper RAG Assistant")
st.write("Upload a biomedical research paper PDF and ask questions based on its content.")

# ── File uploader ──
uploaded_file = st.file_uploader("Upload a PDF research paper", type=["pdf"])


# ════════════════════════════════════════════
# STEP 1: Extract text from PDF
# ════════════════════════════════════════════
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ════════════════════════════════════════════
# STEP 2: Clean and chunk the text
# ════════════════════════════════════════════
def clean_full_text(text):
    """Remove reference/admin sections that add noise to retrieval."""
    cutoff_terms = [
        "Online content", "References", "Acknowledgements",
        "Author contributions", "Competing interests", "Additional information"
    ]
    cleaned_text = text
    for term in cutoff_terms:
        index = cleaned_text.lower().find(term.lower())
        if index != -1:
            cleaned_text = cleaned_text[:index]
    return cleaned_text


def split_text_into_chunks(text):
    cleaned_text = clean_full_text(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(cleaned_text)

    noise_terms = [
        "competing interests", "publisher's note", "reprints and permissions",
        "additional information", "reporting summary", "supplementary information",
        "nature.com", "github.com", "http", "https", "doi.org", "zenodo",
        "references", "data availability", "author contributions"
    ]

    cleaned_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(term in chunk_lower for term in noise_terms):
            continue
        if len(chunk.strip()) < 300:
            continue
        cleaned_chunks.append(chunk)

    return cleaned_chunks


# ════════════════════════════════════════════
# STEP 3: Load embedding model (cached)
# ════════════════════════════════════════════
@st.cache_resource
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# ════════════════════════════════════════════
# STEP 4: Create ChromaDB vector store
# ════════════════════════════════════════════
def create_vector_store(chunks):
    embeddings = load_embedding_model()
    collection_name = f"paper_{uuid.uuid4().hex}"
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    return vector_store


# ════════════════════════════════════════════
# STEP 5: Hybrid retrieval
# ════════════════════════════════════════════
def remove_duplicate_results(results):
    unique_docs = []
    seen_texts = set()
    for doc in results:
        text = doc.page_content.strip()
        signature = text[:300]
        if signature not in seen_texts:
            seen_texts.add(signature)
            unique_docs.append(doc)
    return unique_docs


def keyword_search(chunks, query, top_k=3):
    """Keyword-based fallback search to complement vector search."""
    stopwords = {
        "what", "is", "are", "the", "this", "that", "about", "used",
        "using", "with", "from", "into", "were", "was", "how", "many",
        "does", "do", "in", "on", "of", "to", "a", "an", "and", "or",
        "for", "by", "as", "it", "paper", "research", "article"
    }

    query_words = [
        word.strip(".,?!:;()[]{}")
        for word in query.lower().split()
        if len(word.strip(".,?!:;()[]{}")) > 2
        and word.strip(".,?!:;()[]{}") not in stopwords
    ]

    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        meaningful_phrase = " ".join(query_words)
        if meaningful_phrase and meaningful_phrase in chunk_lower:
            score += 3
        for word in query_words:
            if word in chunk_lower:
                score += 1
        if query_words:
            matched = sum(1 for w in query_words if w in chunk_lower)
            score += matched / len(query_words)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]


def combine_results(vector_results, keyword_results):
    combined = []
    seen = set()
    for text in keyword_results:
        sig = text.strip()[:300]
        if sig not in seen:
            seen.add(sig)
            combined.append(text.strip())
    for doc in vector_results:
        text = doc.page_content.strip()
        sig = text[:300]
        if sig not in seen:
            seen.add(sig)
            combined.append(text)
    return combined


# ════════════════════════════════════════════
# STEP 6: Generate answer using Groq + Llama
# ════════════════════════════════════════════
def generate_answer(question, context_chunks):
    """
    Send the retrieved chunks as context to Llama 3.3 via Groq.
    The LLM reads only the retrieved chunks — not the whole PDF.
    This keeps it fast, cheap on tokens, and accurate.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return "❌ Groq API key not found. Please check your .env file."

    # Join top chunks into one context string
    context = "\n\n---\n\n".join(context_chunks[:4])

    # The prompt — tells the LLM exactly how to behave
    system_prompt = """You are a biomedical research assistant.
Your job is to answer questions based ONLY on the provided research paper context.
Rules:
- Answer clearly and directly based on the context
- If the answer is not in the context, say "I could not find this information in the provided paper."
- Do not make up information
- Keep answers concise but complete
- Use scientific terminology appropriately"""

    user_prompt = f"""Context from the research paper:
{context}

Question: {question}

Answer based on the context above:"""

    try:
        client = Groq(api_key=groq_api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Free Llama 3.3 70B on Groq
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,      # Low temperature = more factual, less creative
            max_tokens=512        # Enough for a thorough answer, not wasteful
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


# ════════════════════════════════════════════
# DISPLAY HELPER
# ════════════════════════════════════════════
def display_chunk_box(text):
    safe_text = html.escape(text)
    st.markdown(
        f"""
        <div style="
            background-color: #1e1e26;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #3a3a45;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            color: #f1f1f1;
        ">
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════
# MAIN APP FLOW
# ════════════════════════════════════════════
if uploaded_file is not None:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Extract
    with st.spinner("📄 Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text.strip():
        with st.expander("📖 Extracted Text Preview"):
            st.write(pdf_text[:2000])
            st.info(f"Total characters extracted: {len(pdf_text)}")

        # Chunk
        with st.spinner("✂️ Cleaning and splitting into chunks..."):
            chunks = split_text_into_chunks(pdf_text)

        st.success(f"✅ {len(chunks)} clean chunks created")

        # Embed + store
        with st.spinner("🔢 Creating embeddings and vector database..."):
            vector_store = create_vector_store(chunks)

        st.success("✅ Vector database ready!")

        # ── Question input ──
        st.markdown("---")
        st.subheader("💬 Ask a Question")
        user_question = st.text_input(
            "Ask anything about this research paper:",
            placeholder="e.g. What trajectory inference methods were benchmarked?"
        )

        if user_question:
            with st.spinner("🔍 Retrieving relevant context..."):
                # Vector search
                vector_results = vector_store.similarity_search(user_question, k=6)
                vector_results = remove_duplicate_results(vector_results)

                # Keyword search
                keyword_results = keyword_search(chunks, user_question, top_k=3)

                # Combine both
                combined = combine_results(vector_results, keyword_results)

            with st.spinner("🧠 Generating answer with Llama 3.3 via Groq..."):
                answer = generate_answer(user_question, combined)

            # ── Show the AI answer prominently ──
            st.markdown("---")
            st.subheader("🤖 AI Answer")
            st.markdown(
                f"""
                <div style="
                    background-color: #0d1f2d;
                    border: 1px solid #1a5276;
                    border-left: 4px solid #2e86c1;
                    border-radius: 8px;
                    padding: 20px;
                    font-size: 15px;
                    line-height: 1.7;
                    color: #d6eaf8;
                ">
                    {html.escape(answer)}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ── Show source evidence below ──
            st.markdown("---")
            st.subheader("📚 Source Evidence")
            st.caption("The AI answer was generated from these retrieved chunks:")

            for i, chunk_text in enumerate(combined[:4]):
                with st.expander(f"Evidence Chunk {i + 1}"):
                    display_chunk_box(chunk_text)

    else:
        st.error("❌ No text could be extracted. This PDF may be scanned or image-based.")

else:
    st.info("👆 Please upload a biomedical research paper PDF to begin.")