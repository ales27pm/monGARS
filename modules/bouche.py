import logging
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class Bouche:
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self.llm = OllamaLLM(model="dolphin-mistral:7b-v2.8-q2_K", base_url="http://localhost:11434")
        self.prompt = PromptTemplate.from_template(
            "You are monGARS, an advanced AI assistant created by Alexis.\n"
            "You respond concisely and naturally in clear, conversational French.\n"
            "Query: {query}\nContext: {context}\nResponse:"
        )

        # Configure logging
        logging.basicConfig(
            filename="logs/mongars.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)

    def respond(self, query):
        try:
            logging.info(f"Bouche: Received query '{query}'")
            if not query.strip():
                logging.warning("Bouche: Query is empty.")
                return "Query is empty. Please provide a valid query."

            context = self.hippocampus.retrieve_memory(query)
            logging.info(f"Bouche: Retrieved context for query '{query}' - {context}")
            if context == "Memory is empty. Add relevant data to the memory before querying.":
                logging.warning("Bouche: No relevant context found for the query.")
                return "No relevant context found. Please refine your query or add data to the memory."

            # Format the prompt into a string
            formatted_prompt = self.prompt.format(query=query, context=context)
            logging.info(f"Bouche: Formatted prompt - {formatted_prompt}")

            # Pass the formatted string directly to the LLM
            response = self.llm.invoke(formatted_prompt)
            logging.info(f"Bouche: Generated response for query '{query}' - {response}")
            return response
        except Exception as e:
            logging.error(f"Bouche: Error responding to query '{query}' - {e}")
            return "Error generating response."
