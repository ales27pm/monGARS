import logging
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_ollama import OllamaLLM

class Cortex:
    def __init__(self):
        self.tasks = []
        self.llm = OllamaLLM(model="dolphin-mistral:7b-v2.8-q5_1", base_url="http://localhost:11434")
        self.prompt = PromptTemplate.from_template("Task: {task}\nExplain the task in detail and suggest next steps.")
        self.chain = RunnableMap({"prompt": self.prompt, "llm": self.llm})

    def add_task(self, task, priority=0):
        self.tasks.append((priority, task))
        self.tasks.sort(reverse=True)
        logging.info(f"Cortex: Task '{task}' added with priority {priority}.")

    def prioritize_tasks(self):
        for i, (priority, task) in enumerate(self.tasks):
            if "urgent" in task.lower():
                self.tasks[i] = (priority + 10, task)
        self.tasks.sort(reverse=True)

    async def process_tasks(self):
        while self.tasks:
            self.prioritize_tasks()
            priority, task = self.tasks.pop(0)
            try:
                logging.info(f"Cortex: Processing task '{task}' with priority {priority}...")
                result = self.chain.invoke(task)
                logging.info(f"Cortex: Task result - {result}")
            except Exception as e:
                logging.error(f"Cortex: Error processing task '{task}' - {e}")
