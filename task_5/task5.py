from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool, StructuredTool, tool
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
import requests
from bs4 import BeautifulSoup

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
    temperature=0.2,
)

# Agent1-Define the Research Agent Prompt
research_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
    You are a highly skilled Market Research Specialist tasked with gathering accurate, up-to-date insights about "{input}" and its industry.

You have access to external tools such as:
search_and_fetch_articles : for real time web searches
Calculator: for numerical estimates

Important Instructions:
Use tools whenever necessary — especially when recent or factual information is needed.
Do NOT guess or hallucinate facts.
-Always cite the source if applicable.

Your research should include the following sections:

1. Company Overview and History 
    Use the Search Tool to find a brief company background.

2.  Market Size and Growth Estimates
    Use Search Tool or Article Fetcher to get recent stats or reports.

3. Top 3 Competitors and Market Shares
    Search for competitor analysis reports or summaries.

4. Current Industry Trends
   Fetch recent articles and summarize 2–3 trends shaping the industry.

5. ain Product/Service Offerings 
   → Identify key products/services listed on official websites or reliable sources.

6. Recent News or Developments (last 6 months)
   → Use the Search Tool to find recent news or press releases.



Respond with a structured report. Use tools as needed before answering each section. Write intermediate thoughts and actions in the scratchpad below to help guide your tool usage:

{agent_scratchpad}
"""
)

# Agent 2 - Define the Analysis Agent Prompt
analysis_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
You are a professional Business Analyst tasked with interpreting structured market research data.

Analyze the following market research content:

{input}


You have access to the following tool(s):
    calculator_tool: Use this only for valid numerical expressions (e.g., 100 * 5 + 2). Do not use for text-based or non-numerical input.

{agent_scratchpad}
"""
)

# Agent 3 - Prompt
report_prompt = PromptTemplate(
    input_variables=["input","topic"],
    template="""
    You are a Professional Report Writer specializing in market research reports.

    Based on the following analysis:
    {input}

    Create a concise market research report on {topic} with these sections:

    1. Executive Summary (brief)
    2. Table of Contents (optional)
    3. Company Overview (short)
    4. Industry Analysis (summarized)
    5. Competitive Landscape (top competitors)
    6. SWOT Analysis (concise)
    7. Growth Opportunities (key opportunities)
    8. Strategic Recommendations (2-3 recommendations)
    9. Conclusion (brief)

    Format the report in professional Markdown with appropriate headers and bullet points. Keep the report under 1000 tokens.
    """
)


@tool
def search_and_fetch_articles(query: str) -> str:
    """
    Searches Google using Serper API and fetches key article content.
    Returns a merged result from 1 top-ranked article.
    """

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}

    try:
        response = requests.post(url, headers=headers,
                                 json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("organic", [])[:1]:  # Only 1 article for performance
            article_url = item.get("link", "")
            content = fetch_article_content(article_url)
            articles.append({"url": article_url, "content": content})

        return concatenate_article_text(articles)

    except Exception as e:
        return f"[Error searching articles]: {str(e)}"


def fetch_article_content(url: str) -> str:
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")

        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text(strip=True)
                           for p in paragraphs[:3]])  # fewer paragraphs
        return content[:1000]  # limit to 1000 characters
    except Exception as e:
        return f"[Error fetching content from {url}]: {str(e)}"



def concatenate_article_text(articles: list) -> str:
    all_text = ""
    for i, article in enumerate(articles, 1):
        all_text += f"\n🔗 Article {i}: {article['url']}\n"
        all_text += article['content'] + "\n\n"
    return all_text.strip()


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a simple math expression using Python eval"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Math error: {e}"

## Defining Agent1
tools = [search_and_fetch_articles,calculator_tool]
# Create the agent
agent1 = create_openai_tools_agent(llm, tools, research_prompt)
agent_executor1 = AgentExecutor(
    agent=agent1,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


def run_agent1(query):
    return agent_executor1.invoke({"input": query})["output"]


## Agent 2 - Analysis chain
agent2_tools = [calculator_tool]
agent2 = create_openai_tools_agent(llm, agent2_tools, analysis_prompt)
agent_executor2 = AgentExecutor(
    agent=agent2,tools=agent2_tools,verbose=True, max_iterations=5)


def run_agent2(research_data):
    research_data = research_data[:12000] # to stay within token limit
    result = agent_executor2.invoke({"input": research_data})
    return result["output"] if isinstance(result, dict) else result


# Agent 3
report_chain = report_prompt|llm

def run_agent3(analysis_results,topic):
    result = report_chain.invoke({"input": analysis_results,"topic":topic})
    return result.content



# worflow 
def run_full_workflow(topic):
    print(f"\n |-> Running Market Research Workflow for: {topic}")

    print("\n Agent 1: Researching...")
    research_data = run_agent1(f"Market research report on {topic}")

    print("\n Agent 2: Analyzing...")
    analysis_output = run_agent2(research_data)

    print("\n Agent 3: Writing Report...")
    final_report = run_agent3(analysis_output,topic)

    with open("market_research_report.md", "w", encoding="utf-8") as f:
        try:
            f.write(str(final_report))
        except Exception as e:
            print(f"[Error writing file]: {e}")

    print("✅ Report saved as 'market_research_report.md'")
    return final_report


# entry point of execution
if __name__ == "__main__":
    run_full_workflow("Tata Motors Limited")

# used large language model is llama provided Groq api