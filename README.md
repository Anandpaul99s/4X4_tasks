## 4X4 Technology Services AI Intern Assignment


## Choosen Tasks - 1,2,3,4,5

## => Folder Structure

![folder](https://github.com/user-attachments/assets/a88c2170-6d9b-4c0b-9ec2-82cc3b17ec58)


## API Key Configuration
	Assignments require external APIs. Create a .env file in the root directory and add the following

		# For Groq LLM Inference (tasks 2,3,4,5)
		GROQ_API_KEY=your_groq_api_key

		# For Google Generative AI Embeddings (tasks 2 and 3)
		GOOGLE_API_KEY=your_google_api_key

		# For Serper.dev Web Search (task 5)
		SERPER_API_KEY=your_serper_api_key


## Get your own api keys from these websites
🔗 Groq: https://console.groq.com
🔗 Google API Key (MakerSuite): https://makersuite.google.com/app/apikey
🔗 Serper.dev API: https://serper.dev
🔗 OpenRouter: https://openrouter.ai

## Models used Groq provided model => Llama3-8b-8192
## For the purpose to generate embeddings => Google Generative AI Embeddings


# Install Dependencies

	Make sure you have Python 3.8 or higher installed.
	Install all required libraries using the following command:

##  command =>  pip install -r requirements.txt


===============================================================================

## Assignment 1: PDF Data Extraction
   File: task_1.py
      Objective: Extract structured data (text, key-value pairs, tables) from PDF documents.
      Libraries: PyMuPDF, pdfplumber, pandas

   How to Run:
   Place 3–5 sample PDFs inside the pdfs/ folder.

## Run:
    python task_1.py
	Extracted output will be saved in:
		output/
		├── tables/         # CSVs of extracted tables
		├── key_values/     # JSON of extracted key-value pairs
		└── text/           # Full extracted text



## Assignment 2: Conversational AI Bot
	File: app.py
		Objective: Create a chatbot that answers questions using uploaded PDFs or text documents.
		Interface: Built with Streamlit
		LLM: Groq (LLaMA), Google Generative Embeddings for vector search

## How to Run:
	
	streamlit run app.py
	
## Features:
   1) Upload .pdf, .docx, or .txt

   2) Builds a knowledge base using FAISS

  3) Chat live with your documents

# API key required : Make sure to configure GROQ_API_KEY and GOOGLE_API_KEY in .env .


## Assignment 3: Retrieval-Augmented Generation (RAG)
	File: task_3.py
	Objective: Implement RAG with FAISS, Google embeddings, and LLMs.

## How to Run:

	Create a folder knowledge_base/ and place .pdf or .txt files.

    1) Run:
		python task_3.py
		
	2) Ask questions in the terminal. Responses will include both the answer and the source chunks.

# API key required : Make sure to configure GROQ_API_KEY and GOOGLE_API_KEY in .env .



## Assignment 4: Document Summarization Engine
	File: task4.py
	Objective: Summarize multi-page PDFs using both extractive and abstractive techniques.
	LLM Used: Llama3-8b-8192 via GROQ

## How to Run:
	python task4.py
	Make sure the file path (inside the script) points to your target .pdf  OR RUN THE SCRIPT PROVIDE FILE PATH IN COMMAND LINE, MAKE SURE THAT FILE IS IN SAME FOLDER.
	
# API key required: Make sure to configure GROQ_API_KEY

## Output:
	summary_output.txt

(I was not able to run this task4.py file because my token limit is exceeding more than its need because i have used map-reduce)


## Assignment 5: AI Agentic Workflow Simulation
	File: task5.py
	Objective: Simulate a 3-agent pipeline to generate a full market research report.
	Technologies: LangChain Agents + Tools + LLM (Groq) + Web scraping

## How to Run:

	python task5.py

## Agent Workflow:

Agent 1: Gathers research via web scraping (Serper + BeautifulSoup)

Agent 2: Analyzes findings using reasoning + calculator tool

Agent 3: Compiles into a professional Markdown report

##  Output:
	market_research_report.md

## API key required: Make sure to configure GROQ_API_KEY and SERPER_API_KEY in .env

=====================================================================================================



