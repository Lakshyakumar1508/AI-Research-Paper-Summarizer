import os
from importlib import import_module
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# ==========================================================
# Compatibility for LangChain
# ==========================================================

try:
    RetrievalQA = import_module("langchain_classic.chains").RetrievalQA
except ImportError:
    RetrievalQA = import_module("langchain.chains").RetrievalQA

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

# ==========================================================
# Configuration
# ==========================================================

PDF_PATH = r"D:\Coding Space\Langchain\Project\Attention Is All You Need.pdf"

FAISS_INDEX = "faiss_index"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ==========================================================
# Load PDF
# ==========================================================

print("=" * 60)
print("Loading PDF...")
print("=" * 60)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages Loaded : {len(documents)}")

# ==========================================================
# Split PDF
# ==========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

docs = text_splitter.split_documents(documents)

print(f"Chunks Created : {len(docs)}")

# ==========================================================
# Gemini Embeddings
# ==========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# ==========================================================
# Create / Load FAISS
# ==========================================================

if os.path.exists(FAISS_INDEX):

    print("\nLoading Existing FAISS Index...")

    vectorstore = FAISS.load_local(
        FAISS_INDEX,
        embeddings,
        allow_dangerous_deserialization=True,
    )

else:

    print("\nGenerating Embeddings...")

    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
    )

    vectorstore.save_local(FAISS_INDEX)

    print("FAISS Index Saved Successfully!")

# ==========================================================
# Retriever
# ==========================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 15,
    },
)

# ==========================================================
# Hugging Face LLM
# ==========================================================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2,
)

# ==========================================================
# Prompt
# ==========================================================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert AI Research Assistant.

Use ONLY the information provided in the context.

If the answer cannot be found, reply:

"I couldn't find this information in the paper."

Always answer clearly, technically, and concisely.

--------------------
Context:
{context}
--------------------

Question:
{question}

Answer:
"""
)

# ==========================================================
# Retrieval QA
# ==========================================================

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt
    },
    return_source_documents=True,
)

# ==========================================================
# Chat Loop
# ==========================================================

print("\n")
print("=" * 60)
print("📚 AI Research Paper Summarizer")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    query = input("\nQuestion: ").strip()

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    try:

        result = qa_chain.invoke(
            {
                "query": query
            }
        )

        print("\nAnswer\n")
        print(result["result"])

        print("\nReferenced Pages")

        pages = set()

        for doc in result["source_documents"]:

            page = doc.metadata.get("page", "Unknown")

            if page not in pages:

                pages.add(page)

                print(f"\nPage {page + 1}")
                print("-" * 50)
                print(doc.page_content[:250].replace("\n", " "))
                print()

    except Exception as e:
        print("\nError:")
        print(e)