
import fitz
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
    temperature=0.2,
)



def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to split text into manageable chunks


def split_text(text, chunk_size=900, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])



map_prompt_template = """Summarize the following text concisely:
{text}
CONCISE SUMMARY:"""

combine_prompt_template = """Combine the following summaries into a comprehensive summary:
{text}
COMPREHENSIVE SUMMARY:"""

MAP_PROMPT = PromptTemplate(
    template=map_prompt_template, input_variables=["text"])
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["text"])



def summarize_document(file_path):
    print("Loading document...")
    text = load_pdf(file_path)

    print("Splitting document into chunks...")
    docs = split_text(text)

    print("Performing extractive summarization...")
    extractive_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=True
    )
    extractive_summary = extractive_chain.run(docs)

    print("Performing abstractive summarization...")
    abstractive_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=True
    )
    abstractive_summary = abstractive_chain.run(docs)

    # Save summaries to files
    with open("summary_output.txt", "w", encoding="utf-8") as f:
        f.write("=== Extractive Summary ===\n\n")
        f.write(extractive_summary + "\n\n")
        f.write("=== Abstractive Summary ===\n\n")
        f.write(abstractive_summary)

    with open("summary_output.md", "w", encoding="utf-8") as f:
        f.write("# =>Document Summary\n\n")
        f.write("## |-> Extractive Summary\n\n")
        f.write(extractive_summary + "\n\n")
        f.write("## |-> Abstractive Summary\n\n")
        f.write(abstractive_summary)

    print("Summaries saved to 'summary_output.txt' and 'summary_output.md'.")



if __name__ == "__main__":
    file_path = input("Enter your file path: ")
    summarize_document(file_path)
