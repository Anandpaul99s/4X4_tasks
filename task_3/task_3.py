import os,fitz
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
load_dotenv()

# Load keys
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


def load_documents_from_folder(folder_path):
    all_text = ""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == ".pdf":
            doc = fitz.open(file_path)
            all_text += "\n".join(page.get_text() for page in doc)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read()
    return all_text


def vector_embeddings(text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore, chunks

def get_llm():
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192",
        temperature=0.2,
    )
    return llm

def build_qa_chain(vectorstore):
    llm = get_llm()
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Use the following pieces of context to answer the question.
        If the answer is not contained in the context, say "I don't know".
        <context>
        {context}
        <context>
        Questions:{question}
        """,
        input_variables=["question","context"]
    )

    chain =  RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain


def run_rag_pipeline(doc_path):
    print(" Loading document...")
    text = load_documents_from_folder(doc_path)

    print(" Creating embeddings and vectorstore...")
    vectorstore, chunks = vector_embeddings(text)

    print("|-> RAG system ready. Ask questions (type 'exit' to quit):")
    qa_chain = build_qa_chain(vectorstore)

    while True:
        user_question = input("\n💬 You: ")
        if user_question.lower() in ["exit", "quit"]:
            print(" Exiting chat.")
            break

        result = qa_chain.invoke({"query": user_question})

        print("\n|-> Answer:")
        print(result['result'])

        print("\n Source Chunks Used:")
        for i, doc in enumerate(result['source_documents']):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content)



if __name__ == "__main__":
    folder_path = "knowledge_base"
    
    run_rag_pipeline(folder_path)
