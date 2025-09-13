# ============================================================
# 1. Install Dependencies
# =============================

# !pip install -qU langchain-openai langgraph langchain
import sqlite3
import os

# =============================
# 2. Setup SQLite Database
# =============================

con = sqlite3.connect("shop.db", check_same_thread=False)

con.execute("""
CREATE TABLE IF NOT EXISTS Users(
    id INTEGER PRIMARY KEY, 
    name TEXT, 
    email TEXT, 
    signup_date DATE
)
""")

con.execute("""
CREATE TABLE IF NOT EXISTS Orders(
    id INTEGER PRIMARY KEY, 
    user_id INTEGER, 
    amount REAL, 
    status TEXT, 
    order_date DATE, 
    FOREIGN KEY(user_id) REFERENCES Users(id)
)
""")

# Insert sample data
con.execute("INSERT INTO Users VALUES(1, 'Jack', 'jack@example.com', '2004/5/1')")
con.execute("INSERT INTO Users VALUES(2, 'Jark', 'jark@example.com', '2004/5/5')")
con.execute("INSERT INTO Orders VALUES(1, 1, 25000.0, 'Pending', '2004/6/2')")
con.execute("INSERT INTO Orders VALUES(2, 2, 35000.0, 'On the way', '2004/7/2')")
con.commit()

# =============================
# 3. Define Tools
# =============================

from langchain.agents import tool
@tool
def get_schema() -> str:
    """Returns table schema from the sqlite database"""
    schema = ""
    for table in ["Users", "Orders"]:
        print(f"\nChecking Table: {table}")
        
        # ✅ FIX: f-string
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        
        print(f"RAW PRAGMA OUTPUT FOR {table}: {rows}")
        columns = [f"{r[1]} {r[2]}" for r in rows]
        cols = ", ".join(columns)
        schema += f"{table}({cols}) "
        
    print("\nFinal Schema:")
    print(schema.strip())
    return schema.strip()

@tool
def execute_sql(query: str) -> str:
    """Execute an SQL query on the connected SQLite database and return the results as a string."""
    try:
        result = con.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error happened: {str(e)}"

# =============================
# 4. Setup LLM Agent
# =============================

from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

tools = [get_schema, execute_sql]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def agent(state: AgentState):
    mes = llm_with_tools.invoke(state["messages"])
    return {"messages": [mes]}

# =============================
# 5. Build LangGraph Workflow
# =============================

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

builder = StateGraph(AgentState)
builder.add_node("Agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("Agent")
builder.add_conditional_edges("Agent", tools_condition)
builder.add_edge("tools", "Agent")

graph = builder.compile()

# Display workflow graph
display(Image(graph.get_graph().draw_mermaid_png()))

# =============================
# 6. Run Queries
# =============================

query1 = {"messages": [{"role": "user", "content": "How many users did not make any purchase please give the user id as well"}]}
result = graph.invoke(query1)
print("\n🔹 Final Answer:")
print(result["messages"][-1].content)

# Streaming Execution
for output in graph.stream(query1):
    for key, value in output.items():
        print(f"\n Node '{key}':")
        print(value["messages"])