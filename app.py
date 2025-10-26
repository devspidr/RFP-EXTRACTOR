
import os
import io
import re
import json
import math
from typing import List, Optional
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from bs4 import BeautifulSoup
import PyPDF2

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ===================== Load Environment =====================
def load_environment():
    """
    Load environment variables from .env file if present.
    Warn the user if GOOGLE_API_KEY is missing.
    """
    load_dotenv(find_dotenv())
    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY not found. Add it to your .env or environment variables.")


# ===================== Optional Dependencies for RAG =====================
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMB_AVAILABLE = True  # Indicates if embeddings & FAISS are available for RAG
except Exception:
    EMB_AVAILABLE = False  # If unavailable, RAG will fallback to keyword scoring


# ===================== Streamlit Config =====================
st.set_page_config(page_title="AI-Powered RFP Extractor", layout="wide")


# ===================== Custom CSS =====================
# This CSS styles the Streamlit app with gradient background, color customization, and rounded buttons
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #b2f2bb, #74c69d); color: #000000; }
h1, h2, h3, h4, h5 { color: #004d26 !important; }
label, .stTextInput label, .stSelectbox label, .stFileUploader label { color: #000000 !important; font-weight: 600 !important; }
.upload-box { background-color: rgba(255,255,255,0.5); padding:25px; border-radius:15px; box-shadow:0 4px 10px rgba(0,0,0,0.15); }
.stDownloadButton>button { background-color:#2d6a4f !important; color:white !important; border-radius:8px !important; border:none !important; }
.stDownloadButton>button:hover { background-color:#40916c !important; transition:0.3s; }
.stTextInput>div>div>input { background-color:#e9f5ec !important; color:#004d26 !important; border-radius:6px !important; }
.stSelectbox>div>div>select { background-color:#e9f5ec !important; color:#004d26 !important; }
.stFileUploader { color:#000000 !important; }
</style>
""", unsafe_allow_html=True)


# ===================== Sidebar =====================
st.sidebar.title("🌿 RFP Extractor - Settings")
st.sidebar.info("Extract structured RFP details using Gemini + LangChain.")
st.sidebar.markdown("""
<div style='color:#e6ffe6'>
    <strong>Built by:</strong> J Soundar Balaji<br>
    <strong>GitHub:</strong> 
    <a href='https://github.com/devspidr' target='_blank' style='color:#90ee90; text-decoration:none;'>devspidr</a>
</div>
""", unsafe_allow_html=True)


# ===================== Page Header =====================
st.title("📑 AI-Powered RFP Extractor — Hybrid NLP + RAG + LLM")
st.markdown("""
**Project Overview:** Extract structured information from RFP documents (PDF/HTML) using AI.  

- **Small documents** (<100 pages for PDFs or <50,000 characters for HTML) → NLP + LLM extraction (**Gemini LLM via LangChain**)  
- **Large documents** (>100 pages for PDFs or >50,000 characters for HTML) → RAG (FAISS VectorStore) + LLM extraction (**SentenceTransformers embeddings + FAISS + Gemini LLM via LangChain**)  

Upload your RFP document below and enter your Gemini API key to start extraction.
""")


# ===================== API Key + Model + Upload =====================
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # User enters Gemini API key
    user_api_key = st.text_input("🔑 Enter Your Gemini API Key", type="password", key="api_key_input", help="Your key is used locally and not stored.")

with col2:
    # User selects which Gemini model to use
    model_choice = st.selectbox(
        "🤖 Select Gemini Model",
        ["gemini-2.5-flash","gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-pro","gemini-pro"],
        index=0,
        key="model_choice_select"
    )

# Upload RFP document (PDF or HTML)
uploaded_file = st.file_uploader(
    "📂 Upload RFP document", type=["pdf","html","htm"], key="rfp_file_upload",
    help="Supports PDF/HTML. Only one file at a time."
)
st.markdown('</div>', unsafe_allow_html=True)

# Stop execution if API key is missing
if uploaded_file and not user_api_key:
    st.warning("⚠️ Please enter your Gemini API key to start extraction.")
    st.stop()


# ===================== Helper Functions =====================

def parse_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file.
    Returns concatenated text from all pages.
    """
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        st.warning(f"PDF parse failed: {e}")
        return ""


def parse_html_bytes(file_bytes: bytes) -> str:
    """
    Extract clean text from an HTML file.
    Removes scripts, styles, and metadata.
    """
    try:
        soup = BeautifulSoup(file_bytes.decode("utf-8", errors="ignore"), "html.parser")
        for t in soup(["script","style","noscript","meta"]):
            t.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Break large text into smaller chunks with optional overlap.
    Tries to end chunks at sentence boundaries.
    """
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:].strip())
            break
        window_end = min(length, end + 200)
        segment = text[start:window_end]
        m = list(re.finditer(r"[.!?]\s", segment))
        if m:
            end = start + m[-1].end()
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
    return chunks


# ===================== Constants =====================
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 4

TARGET_FIELDS = [
    "Bid Number","Title","Due Date","Bid Submission Type","Term of Bid",
    "Pre Bid Meeting","Installation","Bid Bond Requirement","Delivery Date",
    "Payment Terms","Any Additional Documentation Required","MFG for Registration",
    "Contract or Cooperative to use","Model_no","Part_no","Product",
    "Contact Info","Company Name","Bid Summary","Product Specification"
]


# ===================== Vector Index Class for RAG =====================
class VectorIndex:
    """
    Builds FAISS vector index for chunked text.
    Supports searching top-k relevant chunks for a query.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if not EMB_AVAILABLE:
            raise RuntimeError("Embeddings or FAISS not installed.")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.id_to_chunk = {}
        self.next_id = 0

    def add(self, chunks: List[str]):
        """Add text chunks to the FAISS index."""
        embs = self.model.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        self.index.add(embs.astype("float32"))
        for ch in chunks:
            self.id_to_chunk[self.next_id] = ch
            self.next_id += 1

    def search(self, query: str, top_k: int = TOP_K):
        """Search the FAISS index for top-k relevant chunks."""
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype("float32"), top_k)
        return [(self.id_to_chunk.get(int(idx)), float(score)) for score, idx in zip(D[0], I[0]) if idx >= 0]


# ===================== Keyword Score Fallback =====================
def keyword_score_retriever(chunks: List[str], query: str, top_k: int = TOP_K):
    """
    Fallback retrieval method if FAISS is unavailable.
    Scores chunks based on keyword overlap with query.
    """
    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    scored = []
    for ch in chunks:
        ch_tokens = re.findall(r"\w+", ch.lower())
        overlap = sum(1 for t in q_tokens if t in ch_tokens)
        score = overlap / (1 + math.log(1 + len(ch_tokens)))
        scored.append((ch, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ===================== LLM Extractor =====================
class LLMExtractor:
    """
    Wraps Gemini LLM to extract fields from text chunks.
    Returns JSON with field-value pairs.
    """
    def __init__(self, model):
        self.llm = model

    def extract_field_json(self, field: str, context_chunks: List[str]):
        """
        Extracts a single field from a list of context chunks.
        Returns value or None.
        """
        joined = "\n\n---\n\n".join(context_chunks)
        prompt = f"""
You are an accurate AI assistant. From the text provided, extract the fields exactly as listed.
If a value is missing, provide a reasonable inference or 'Null'.
Field: "{field}"
Context:
{joined}
Output a JSON in the format: {{ "{field}": "value" }}
"""
        try:
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp)).strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m: return None
            parsed = json.loads(m.group(0))
            return parsed.get(field)
        except Exception:
            return None


# ===================== Normalizers =====================
def normalize_date(s: Optional[str]):
    """Parse and normalize date strings into ISO format."""
    if not s: return None
    from dateutil.parser import parse
    try:
        s_clean = re.sub(r"\b(on|by|due)\b[:\s]*", "", s, flags=re.I)
        return parse(s_clean, fuzzy=True).date().isoformat()
    except Exception:
        return s


def normalize_contact(s: Optional[str]):
    """Extract emails and phone numbers from contact info."""
    if not s: return None
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", s)
    phones = re.findall(r"\+?\d[\d\-\s()]{6,}\d", s)
    return {"raw": s, "emails": list(set(emails)), "phones": list(set(phones))}


# ===================== Main Extraction Logic =====================
load_environment()

if uploaded_file and user_api_key:
    # Set API key and initialize LLM model
    os.environ["GOOGLE_API_KEY"] = user_api_key
    model = ChatGoogleGenerativeAI(model=model_choice)
    extractor = LLMExtractor(model)

    filename = uploaded_file.name
    st.info(f"📂 File: {filename}")
    file_bytes = uploaded_file.getvalue()

    # Extract text depending on file type
    ext = filename.split(".")[-1].lower()
    text = parse_pdf_bytes(file_bytes) if ext=="pdf" else parse_html_bytes(file_bytes)

    if not text.strip():
        st.error("No text extracted from file.")
        st.stop()

    # ------------------ Decide extraction method ------------------
    use_rag = False
    doc_info = ""
    if ext == "pdf":
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
            if num_pages > 100:   # RAG threshold for PDFs
                use_rag = True
            doc_info = f"{num_pages} pages"
        except:
            doc_info = f"{len(text)//2000} pages approx"
            if len(text) > 50000: 
                use_rag = True
    else:  # HTML
        chars = len(text)
        doc_info = f"{chars} characters"
        if chars > 50000: 
            use_rag = True

    st.markdown(f"**ℹ️ Document is {doc_info} — Using {'RAG + LLM' if use_rag else 'NLP + LLM'} for extraction**")


# ===================== NLP + LLM extraction =====================
def extract_nlp_llm(text: str) -> dict:
    """
    Directly use LLM to extract fields from small documents.
    """
    results = {}
    for field in TARGET_FIELDS:
        value = extractor.extract_field_json(field, [text])
        if "date" in field.lower():
            value = normalize_date(value)
        elif "contact" in field.lower() or "info" in field.lower():
            value = normalize_contact(value)
        results[field] = value or "N/A"
    return results


# ===================== RAG + LLM extraction =====================
def extract_rag_llm(text: str) -> dict:
    """
    Use chunking + FAISS index (or keyword fallback) for extracting fields
    from large documents.
    """
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    extracted = {}

    index = None
    if EMB_AVAILABLE:
        with st.spinner("Building FAISS index for RAG..."):
            try:
                index = VectorIndex()
                index.add(chunks)
                st.success("FAISS index ready ✅")
            except Exception as e:
                st.warning(f"FAISS setup failed: {e}")
                index = None

    progress = st.progress(0)
    total = len(TARGET_FIELDS)

    for i, field in enumerate(TARGET_FIELDS):
        if index:
            retrieved = [c for c, _ in index.search(field, top_k=TOP_K)]
        else:
            retrieved = [c for c, _ in keyword_score_retriever(chunks, field, top_k=TOP_K)]

        if not retrieved:
            fallback = [ch for ch in chunks if re.search(field, ch, re.I)]
            retrieved = fallback[:TOP_K] if fallback else chunks[:TOP_K]

        value = extractor.extract_field_json(field, retrieved)
        if "date" in field.lower():
            value = normalize_date(value)
        elif "contact" in field.lower() or "info" in field.lower():
            value = normalize_contact(value)
        extracted[field] = value or "Null"
        progress.progress((i + 1) / total)

    return extracted


# ===================== Run Extraction =====================
results = {}
if uploaded_file and user_api_key:
    if use_rag:
        results[filename] = extract_rag_llm(text)
    else:
        results[filename] = extract_nlp_llm(text)

    st.success("✅ Extraction complete!")

    # Display JSON
    st.subheader("Extracted JSON")
    st.json(results[filename])

    # Download Button
    # ------------------ Save JSON automatically to outputs/ folder ------------------
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_extracted.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results[filename], f, indent=4)

    st.success(f"✅ JSON automatically saved to {output_path}")

