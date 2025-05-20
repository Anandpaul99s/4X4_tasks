# chat with multiple pdf's
import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

LLM = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
    temperature=0.2,
)

EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# defining the prompt
PROMPT = PromptTemplate.from_template(
    """
You are a helpful assistant. Use the following pieces of context to answer the question.
If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:
"""
)


st.set_page_config(page_title="RAG Chat", page_icon="🗂️", layout="wide")
st.title("📄🔍 RAG Chat over Your Documents")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    build_vectors = st.button("🔨 Build Knowledge‑Base")

# maintaining session state here
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Document loading and splitting,injecting into vectorDB
def load_and_split(files):
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    for file in files:
        suffix = file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        if suffix == "pdf":
            pages = PyPDFLoader(tmp_path).load_and_split()
            docs.extend(pages)
        elif suffix == "docx":
            docs.extend(Docx2txtLoader(tmp_path).load_and_split())
        elif suffix == "txt":
            docs.extend(TextLoader(tmp_path).load_and_split())
        else:
            st.warning(f"Skipping unsupported file type: {file.name}")

    return splitter.split_documents(docs)


# After building Documents and splitting those are stored converted in to embedding and stored in VectorDB
# VectorDB - used is FAISS
if build_vectors:
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Embedding and indexing…"):
            documents = load_and_split(uploaded_files)
            st.session_state.vector_store = FAISS.from_documents(
                documents, EMBEDDINGS
            )
        st.success("✅ Knowledge‑base ready!")


st.divider()
st.header("💬 Chat with your documents")

prompt_user = st.chat_input(
    "Ask a question...",
    disabled=st.session_state.vector_store is None,
)


def rag_answer(query):
    # Ensure we have a valid vector store
    if st.session_state.vector_store is None:
        return {"answer": "No knowledge base found."}

    retriever = st.session_state.vector_store.as_retriever()
    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    final_prompt = PROMPT.format(context=context, question=query)

    answer = LLM.invoke(final_prompt)

    return {
        "answer": answer.content,
        "context": relevant_docs,
    }


if prompt_user:
    with st.spinner("Thinking…"):
        start = time.time()
        result = rag_answer(prompt_user)
        elapsed = time.time() - start

    answer = result["answer"]
    context_docs = result["context"]

    # Display Q‑A
    st.session_state.chat_history.append(("user", prompt_user))
    st.session_state.chat_history.append(("ai", answer))

    for role, msg in st.session_state.chat_history:
        align = "user" if role == "user" else "assistant"
        st.chat_message(align).write(msg)

    

    st.caption(f"⏱️ Response time: {elapsed:.2f}s")
