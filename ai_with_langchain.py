from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
tool = TavilySearchResults(max_results=2)
print("------------------")
print(type(tool))
print(tool.name)
print("------------------")
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_groqai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exist_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = { t.name: t for t in tools }
        self.model = model.bind_tools([get_tavily_schema()])
    
    def exist_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_groqai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        print("Raw tool calls from model:", tool_calls)
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
def get_tavily_schema():
    return {
        "type": "function",
        "function": {
            "name": "tavily_search_results_json",
            "description": "Searches the web for up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }

prompt="""You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model=ChatGroq(model="mixtral-8x7b-32768")
abot = Agent(model, [tool], system=prompt)
# abot.graph.get_graph().draw_png("graph.png")
# print("Graph saved to graph.png")

messages = [HumanMessage(content="Who won the super bowl in 2024?")]
result = abot.graph.invoke({"messages": messages})
print('result:')
# print("\\n" + str(result));
print("-------------------------")
print(result['messages'][-1].content)
print("-------------------------")