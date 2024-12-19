import os
import logging
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class Hippocampus:
    def __init__(self):
        # Ensure the logs directory exists
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure logging
        logging.basicConfig(
            filename=os.path.join(log_dir, "mongars.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)

        # Initialize database connection and embeddings
        self.conn = psycopg2.connect(
            dbname="hippocampus", user="app_user", password="1406", host="localhost", port=5432
        )
        self.cursor = self.conn.cursor()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.hierarchical_index = {}

    def store_memory(self, key, value):
        try:
            timestamp = datetime.now()
            embedding = self.embeddings.embed_query(value)
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts([value], self.embeddings, metadatas=[{"key": key, "timestamp": str(timestamp)}])
            else:
                self.vector_store.add_texts([value], metadatas=[{"key": key, "timestamp": str(timestamp)}])

            self.cursor.execute(
                """
                INSERT INTO memory (key, value, vector, timestamp)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, timestamp = EXCLUDED.timestamp
                """,
                (key, Json(value), embedding, timestamp),
            )
            category = self._categorize_memory(value)
            if category not in self.hierarchical_index:
                self.hierarchical_index[category] = []
            self.hierarchical_index[category].append(key)

            self.conn.commit()
            logging.info(f"Hippocampus: Stored memory - {key} at {timestamp} under category '{category}'")
        except Exception as e:
            logging.error(f"Hippocampus: Error storing memory - {e}")

    def retrieve_memory(self, query):
        try:
            if not self.vector_store:
                logging.warning("Hippocampus: Memory is empty. Add relevant data before querying.")
                return "Memory is empty. Add relevant data to the memory before querying."
            retriever = self.vector_store.as_retriever()
            result = retriever.get_relevant_documents(query)
            if result:
                logging.info(f"Hippocampus: Retrieved memory for query '{query}'")
                return result[0].page_content
            else:
                logging.info(f"Hippocampus: No relevant memory found for query '{query}'")
                return "No relevant memory found."
        except Exception as e:
            logging.error(f"Hippocampus: Error retrieving memory - {e}")
            return "Error retrieving memory."

    def retrieve_by_category(self, category):
        try:
            if category not in self.hierarchical_index:
                logging.info(f"Hippocampus: No memories found under category '{category}'")
                return f"No memories found under category '{category}'."
            memory_keys = self.hierarchical_index[category]
            results = []
            for key in memory_keys:
                self.cursor.execute("SELECT value FROM memory WHERE key = %s", (key,))
                record = self.cursor.fetchone()
                if record:
                    results.append(record[0])
            if results:
                logging.info(f"Hippocampus: Retrieved {len(results)} memories under category '{category}'")
                return results
            else:
                logging.info(f"Hippocampus: No detailed memories found for category '{category}'")
                return f"No detailed memories found for category '{category}'."
        except Exception as e:
            logging.error(f"Hippocampus: Error retrieving by category '{category}' - {e}")
            return f"Error retrieving category '{category}'."

    def _categorize_memory(self, value):
        if "project" in value.lower():
            return "Projects"
        elif "research" in value.lower():
            return "Research"
        elif "task" in value.lower():
            return "Tasks"
        else:
            return "General"

    def list_all_memories(self):
        try:
            self.cursor.execute("SELECT key, value FROM memory")
            records = self.cursor.fetchall()
            memories = {record[0]: record[1] for record in records}
            logging.info("Hippocampus: Listed all memories.")
            return memories
        except Exception as e:
            logging.error(f"Hippocampus: Error listing all memories - {e}")
            return "Error listing all memories."
