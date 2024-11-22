
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# In-memory storage for simplicity in testing
memories = []
connections = []
memory_id_counter = 1

class MemoryInput(BaseModel):
    content: str
    metadata: str
    created_at: str

class ConnectionInput(BaseModel):
    source_id: int
    target_id: int
    relationship: str

@app.get("/health/")
def health_check():
    return {"status": "OK"}

@app.post("/memories/")
def add_memory(memory: MemoryInput):
    global memory_id_counter
    try:
        memory_entry = {
            "id": memory_id_counter,
            "content": memory.content,
            "metadata": memory.metadata,
            "created_at": memory.created_at,
        }
        memories.append(memory_entry)
        memory_id_counter += 1
        return {"message": "Memory added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

@app.get("/memories/{memory_id}")
def get_memory(memory_id: int):
    memory = next((m for m in memories if m["id"] == memory_id), None)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found.")
    memory_connections = [c for c in connections if c["source_id"] == memory_id]
    return {"memory": memory, "connections": memory_connections}

@app.post("/connections/")
def connect_memories(connection: ConnectionInput):
    source = next((m for m in memories if m["id"] == connection.source_id), None)
    target = next((m for m in memories if m["id"] == connection.target_id), None)
    if not source or not target:
        raise HTTPException(status_code=404, detail="Source or target memory not found.")
    connection_entry = {
        "source_id": connection.source_id,
        "target_id": connection.target_id,
        "relationship": connection.relationship,
    }
    connections.append(connection_entry)
    return {"message": "Connection added successfully."}
