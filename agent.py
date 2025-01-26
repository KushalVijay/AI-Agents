from phi.agent import Agent
from dotenv import load_dotenv
from phi.model.groq import Groq
from phi.model.deepseek import DeepSeekChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.wikipedia import WikipediaTools

load_dotenv()

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

wikipedia_agent = Agent(
    name="Wikipedia Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[WikipediaTools()],
    instructions=["Use Url and Title to display data"],
    markdown=True,
    debug_mode=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    markdown=True,
    debug_mode=True,
)

agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent, wikipedia_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

agent_team.print_response("Summarize analyst recommendation around OpenAI also get the latest news around this topic. I also want a wikipedia article to understand it in detail.", stream=True)
