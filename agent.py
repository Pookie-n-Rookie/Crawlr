import os
from dotenv import load_dotenv
from tavily import TavilyClient
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")  # Make sure this is correct in your .env

# ------------------- Web Search Tool -------------------
class SearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Searches the web for recent and relevant information."

    def _run(self, query: str) -> str:
        try:
            if not TAVILY_API_KEY:
                raise ValueError("Tavily API key not found")
                
            tav = TavilyClient(api_key=TAVILY_API_KEY)
            result = tav.search(
                query=query,
                include_answers=True,
                max_results=5,
                search_depth="advanced"
            )
            links = [f"{i+1}. {item['title']}\n   {item['url']}"
                    for i, item in enumerate(result['results'][:3])]
            answer = result.get('answer', 'I found these resources:')
            return f"{answer}\n\nTop Links:\n" + "\n\n".join(links)
        except Exception as e:
            return f"⚠️ Search failed. Error: {str(e)}"

# ------------------- LLM Configuration -------------------
def set_llm() -> ChatOpenAI:
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not found in environment variables")
    
    # Ensure the API key is properly stripped of whitespace
    groq_api_key = GROQ_API_KEY.strip()
    
    return ChatOpenAI(
        model="groq/llama3-70b-8192",
        api_key=groq_api_key,  # Use the stripped key
        base_url="https://api.groq.com/openai/v1",
        temperature=0.5,
        max_tokens=1024
    )

# ------------------- Crew Setup -------------------
def form_crew(query: str) -> Crew:
    llm = set_llm()
    researcher = Agent(
        role="AI Research Specialist",
        goal="Deliver precise and informative answers with credible sources",
        backstory=(
            "You're an elite researcher who uses advanced tools to find and "
            "synthesize information from verified web sources."
        ),
        tools=[SearchTool()],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    task = Task(
        description=(
            f"Research the topic: '{query}'\n"
            "Instructions:\n"
            "- Only use verified, recent information\n"
            "- Present a clear and concise summary\n"
            "- Highlight key aspects\n"
            "- Provide 3–5 relevant sources with titles and URLs"
        ),
        expected_output=(
            "1. Overview of the topic\n"
            "2. Key findings\n"
            "3. Summary with evidence\n"
            "4. 3–5 Reference links with titles"
        ),
        agent=researcher,
        output_file="research_output.md"
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
        process="sequential"
    )

# ------------------- Main Runner -------------------
def main() -> None:
    print("🤖 AI Research Assistant (Ctrl+C or 'exit' to stop)\n")
    while True:
        try:
            query = input("🔎 What would you like to research? ")
            if query.strip().lower() in ['exit', 'quit']:
                break

            print("\n🧠 Researching... please wait...\n")
            crew = form_crew(query)
            result = crew.kickoff()

            print("\n✅ Research Result:\n")
            print(result)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in .env file")
    elif not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in .env file")
    else:
        main()