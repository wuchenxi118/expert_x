from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def multiplication_calculator(number1:str,number2:str,reason:str)->str:
    """计算两个数字的乘积,reason中填写调用此工具的原因"""
    return str(float(number1)*float(number2))

# tools = [multiplication_calculator]
# llm = ChatOpenAI(model="anything",base_url="http://192.168.3.32:8085",openai_api_key='anything')
llm = ChatOllama(model="llama3.1",num_ctx=4096,num_predict=1024,temperature=0)
llm_with_tools = llm.bind_tools([multiplication_calculator],tool_choice="auto")


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[multiplication_calculator])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print(value)
            # if isinstance(value["messages"][-1], BaseMessage):
            #     print("Assistant:", value["messages"][-1].content)