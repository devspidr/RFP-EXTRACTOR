

# 📑 AI-Powered RFP Extractor — Hybrid NLP + RAG + LLM (Gemini + LangChain)

An intelligent Streamlit application that automatically extracts **structured data** from **Request for Proposal (RFP)** documents in **PDF** or **HTML** format.  
It uses **Google Gemini (via LangChain)** for text understanding and combines **NLP-based extraction** with a **RAG (Retrieval-Augmented Generation)** pipeline for large documents.

---

## 🚀 Features

✅ Upload RFPs in **PDF** or **HTML** format  
✅ Supports both **small** and **large** documents  
✅ Dual Extraction Mode: **NLP + LLM** (for small docs) and **RAG + LLM** (for large docs)  
✅ Extracts 20+ key RFP fields (Bid No., Due Date, Vendor Info, Terms, etc.)  
✅ Automatic saving of extracted JSON to `outputs/` folder  
✅ Interactive and beautiful **Streamlit UI** with gradient theme  
✅ Supports **Gemini Flash**, **Pro**, and **1.5 / 2.0** variants  
✅ Fallbacks for missing dependencies (FAISS, Sentence Transformers)  

---

## 🧠 How It Works

The application uses **two distinct modes** of extraction based on **document size**:



### 1️⃣ NLP + LLM Mode (for small documents)
- Used when PDF has **≤100 pages** or HTML has **≤50,000 characters**.
- The **entire text** of the document is sent directly to **Gemini LLM**.
- The LLM extracts predefined fields using structured prompts.
- Quick and efficient for short RFPs.

> **Pipeline:**
> PDF/HTML → Text Extraction → Gemini Prompt → JSON Output

---

### 2️⃣ RAG + LLM Mode (for large documents)
- Used when PDF has **>100 pages** or HTML has **>50,000 characters**.
- The document is split into **smaller overlapping chunks**.
- Each chunk is embedded using **Sentence Transformers** (`all-MiniLM-L6-v2`).
- Embeddings are stored in a **FAISS vector index** for fast retrieval.
- For each field (e.g., "Due Date"), the top relevant chunks are retrieved and passed to **Gemini** for field extraction.

> **Pipeline:**
> PDF/HTML → Chunking → Embedding → FAISS Search → Gemini Prompt per Field → JSON Output

If FAISS or Sentence Transformers aren’t installed, a **keyword-based fallback retriever** is automatically used.

---


FOLDER STRUCTURE
```
RFP-EXTRACTOR/
│
├── .devcontainer/            # Dev Container configuration
├── outputs/                  # Folder to store extracted JSON files
├── .gitignore                # Git ignore rules
├── Addendum 1 RFP JA-207652 Student and Staff Computing Devices.pdf
├── Student and Staff Computing Devices SOURCING #168884 - Bid Information - {3} _ BidNet Direct.html
├── README.md                 # Project documentation
├── app.py                    # Main Streamlit application
└── requirements.txt          # List of required Python dependencies
```

## ⚙️ Key Components

| Component | Purpose |
|------------|----------|
| **PyPDF2** | Extracts text from PDF files |
| **BeautifulSoup** | Extracts clean text from HTML |
| **LangChain + Gemini** | Handles LLM-based extraction |
| **SentenceTransformers** | Creates embeddings for RAG |
| **FAISS** | Enables fast semantic retrieval |
| **Streamlit** | Interactive web-based UI |
| **dotenv** | Manages environment variables |

---

## 📋 Extracted Fields

The app intelligently extracts structured information such as:

- Bid Number  
- Title  
- Due Date  
- Bid Submission Type  
- Term of Bid  
- Pre-Bid Meeting  
- Installation  
- Bid Bond Requirement  
- Delivery Date  
- Payment Terms  
- Additional Documentation Required  
- MFG for Registration  
- Contract/Cooperative to Use  
- Model Number  
- Part Number  
- Product  
- Contact Info  
- Company Name  
- Bid Summary  
- Product Specification  

---

## 🧩 Step-by-Step Workflow

### **1️⃣ Launch the Application**
```bash
streamlit run app.py
```

2️⃣ Enter API Key

Enter your Gemini API key in the input box.
(Stored locally and never saved.)

3️⃣ Upload RFP File

Upload either:
```
    .pdf file

    .html or .htm file
```

4️⃣ The App Automatically Chooses Extraction Mode
```
## 📄 Document Extraction Types

| File Type         | Size Criteria       | Extraction Method |
| ----------------- | ----------------- | ---------------- |
| PDF               | ≤ 100 pages        | NLP + LLM        |
| PDF               | > 100 pages        | RAG + LLM        |
| HTML              | ≤ 50,000 characters | NLP + LLM      |
| HTML              | > 50,000 characters | RAG + LLM      |

```

5️⃣ Extraction Process

    Small: Single Gemini call with entire text.

    Large: Chunking → Vector Index → Field-wise Gemini calls.

6️⃣ Results Display

    Structured JSON appears in the app.

    File is auto-saved to:

    outputs/<filename>_extracted.json

7️⃣ Download or Review

You can view, copy, or download the structured JSON.







# ----------------------------------------------------------------------------------------




# INSTALLATION 




# ----------------------------------------------------------------------------------------



1️. Clone the Repository
```
git clone https://github.com/devspidr/RFP-EXTRACTOR.git
cd RFP-EXTRACTOR
```

2️. Create and Activate Virtual Environment
```
python -m venv venv
venv\Scripts\activate   # (Windows)
source venv/bin/activate  # (Mac/Linux)
```

3️. Install Dependencies
```
pip install -r requirements.txt
```

4️. Add Your Google Gemini API Key

Create a .env file:
```
GOOGLE_API_KEY=your_api_key_here
```

5.Run the Streamlit app (assuming your main file is app.py):

```
streamlit run app.py
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

6.Enter your gemini-api-key and select whichever model you need








🧪 Example Output

Example extracted JSON:
```
{
    "Bid Number": "JA-207652",
    "Title": "Student and Staff Computing Devices",
    "Due Date": "2025-04-15",
    "Bid Submission Type": "Online Submission",
    "Term of Bid": "3 years",
    "Pre Bid Meeting": "2025-03-22",
    "Installation": "Yes",
    "Bid Bond Requirement": "5%",
    "Delivery Date": "2025-05-10",
    "Payment Terms": "Net 30 days",
    "Contact Info": {
        "raw": "Procurement Office: bids@district.edu, (555) 123-4567",
        "emails": ["bids@district.edu"],
        "phones": ["(555) 123-4567"]
    }
}
```


🛠️ Tech Stack
| Layer             | Technology                |
| ----------------- | ------------------------- |
| **Frontend/UI**   | Streamlit                 |
| **Backend/LLM**   | LangChain + Google Gemini |
| **Embeddings**    | Sentence Transformers     |
| **Vector Search** | FAISS                     |
| **Parsing**       | PyPDF2, BeautifulSoup     |
| **Environment**   | dotenv                    |


💡 Notes & Best Practices

    Keep RFP documents cleanly formatted for best accuracy.

    Gemini Flash models are faster, while Pro models provide higher accuracy.

    Large RFPs are automatically handled via RAG pipeline.

    The app does not store your uploaded files or API keys permanently.

👨‍💻 Developed By

J Soundar Balaji

