from langgraph.graph import StateGraph, END, START
from app.models import BotState
from app.chatbot.nodes import GraphNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow."""
    
    def __init__(self, nodes: GraphNodes):
        self.nodes = nodes
        self.graph = self.build_graph()
    
    def build_graph(self):
        """Construct the LangGraph workflow."""
        workflow = StateGraph(BotState)

        # Add nodes
        workflow.add_node("greeting", self.nodes.greeting_node)
        workflow.add_node("faq", self.nodes.faq_node)
        workflow.add_node("summarize", self.nodes.summarize_node)

        # Add conditional edges from START
        workflow.add_conditional_edges(
            START,
            self.nodes.classify_intent_node,
            {
                "greeting": "greeting",
                "faq": "faq",
                "summarize": "summarize",
            },
        )

        # Add edges to END
        workflow.add_edge("greeting", END)
        workflow.add_edge("faq", END)
        workflow.add_edge("summarize", END)

        return workflow.compile()
    
    def invoke(self, user_query: str) -> str:
        """Invoke the graph with a user query."""
        in_state = {"user_query": user_query, "response": ""}
        result = self.graph.invoke(in_state)
        return result.get("response", "[ERROR] No response generated.")