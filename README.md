# 📚 AI Research Paper Summarizer (RAG)

An AI-powered **Research Paper Summarizer** built using **LangChain**, **Google Gemini Embeddings**, **FAISS**, and a **Hugging Face Large Language Model**. The application enables users to interactively ask questions about any research paper in PDF format and receive context-aware answers using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- 📄 Load and process research papers in PDF format
- ✂️ Intelligent text chunking using Recursive Character Text Splitter
- 🧠 Semantic embeddings generated with Google Gemini Embeddings
- 🔍 Fast similarity search using FAISS Vector Store
- 🤖 Question Answering using Hugging Face Large Language Model
- 📚 Retrieval-Augmented Generation (RAG)
- 🎯 Context-aware responses grounded in the research paper
- 📑 Displays source pages used to generate the answer
- 💾 Saves FAISS index locally to avoid regenerating embeddings
- 💻 Interactive Command Line Interface (CLI)

---

## 🏗️ Project Architecture

```
PDF Research Paper
        │
        ▼
PyPDFLoader
        │
        ▼
Text Chunking
(RecursiveCharacterTextSplitter)
        │
        ▼
Gemini Embeddings
(gemini-embedding-001)
        │
        ▼
FAISS Vector Database
        │
        ▼
Retriever (MMR Search)
        │
        ▼
Prompt Template
        │
        ▼
Hugging Face LLM
        │
        ▼
Final Answer
```

---

## 🛠️ Technologies Used

- Python
- LangChain
- Google Gemini Embeddings
- Hugging Face Inference API
- FAISS
- PyPDF
- dotenv

---

## 📂 Project Structure

```
AI_Research_Summarizer/
│
├── project/
│   ├── AI_Research_Summarizer.py
│   └── Attention Is All You Need.pdf
│
├── faiss_index/
│
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Research-Paper-Summarizer.git

cd AI-Research-Paper-Summarizer
```

---

### Create Virtual Environment

**Windows**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```text
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

---

## ▶️ Run the Project

```bash
python project/AI_Research_Summarizer.py
```

---

## 💬 Example

```
Question:
What is the main contribution of this paper?

Answer:

The paper introduces the Transformer architecture, a sequence-to-sequence model based entirely on attention mechanisms, eliminating the need for recurrent and convolutional neural networks while achieving state-of-the-art performance on machine translation tasks.

Referenced Pages

Page 2
Page 3
```

---



---

## ⚡ Retrieval Configuration

| Parameter | Value |
|-----------|------:|
| Chunk Size | 1200 |
| Chunk Overlap | 200 |
| Search Type | MMR |
| Top K Results | 4 |
| Fetch K | 15 |

---

## 🧠 Models Used

### Embedding Model

- Google Gemini Embedding
  - `models/gemini-embedding-001`

### Language Model

- Hugging Face Inference API
- Qwen / Mistral / Gemma (configurable)

---

## 📈 Workflow

1. Load PDF
2. Extract text
3. Split into semantic chunks
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant chunks
7. Send retrieved context to LLM
8. Generate context-aware answer
9. Display referenced source pages

---

## 🎯 Future Improvements

- Multi-PDF support
- Research paper comparison
- Automatic paper download from arXiv
- Chat history
- Citation generation
- PDF highlighting
- Export answers to Markdown/PDF
- Support for multiple embedding models

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Lakshya Kumar**

B.Tech Computer Science (AI & ML)

Haridwar University

---

## ⭐ If you found this project useful, consider giving it a star!
