from langchain.agents import Tool

def get_tools(hippocampus, bouche):
    return [
        Tool(name="Memory Retrieval", func=hippocampus.retrieve_memory, description="Retrieve stored memories."),
        Tool(name="Retrieve by Category", func=hippocampus.retrieve_by_category, description="Retrieve memories by category."),
        Tool(name="Respond to Query", func=bouche.respond, description="Generate a response to user queries."),
    ]
