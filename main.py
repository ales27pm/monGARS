import asyncio
from aiohttp import web
import logging
from modules.cortex import Cortex
from modules.hippocampus import Hippocampus
from modules.bouche import Bouche
from modules.document_processor import DocumentProcessor

PORT = 8081

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def start_server():
    # Initialisation des modules
    hippocampus = Hippocampus()
    cortex = Cortex()
    bouche = Bouche(hippocampus)
    document_processor = DocumentProcessor("uploads")

    # Ajouter des mémoires initiales dans Hippocampus
    hippocampus.store_memory("Identity", "Bonjour Alexis, vous êtes mon créateur et maître.")
    hippocampus.store_memory("Greeting", "Bonjour, je suis monGARS, votre assistant intelligent.")

    @web.middleware
    async def error_middleware(request, handler):
        try:
            return await asyncio.wait_for(handler(request), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"Request timeout on {request.path}")
            return web.json_response({"error": "Request timeout"}, status=504)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)

    # Routes de l'API
    async def bouche_respond(request):
        try:
            if not request.can_read_body:
                return web.json_response({"error": "No body provided"}, status=400)
            
            data = await request.json()
            query = data.get("query", "").strip()
            
            if not query:
                logger.info("Empty query received")
                return web.json_response({"error": "Empty query"}, status=400)
            
            logger.info(f"Processing query: {query}")
            response = await asyncio.shield(bouche.respond(query))
            logger.info(f"Response generated: {response}")
            return web.json_response({"response": response})
            
        except Exception as e:
            logger.error(f"Error in bouche_respond: {str(e)}")
            raise

    async def cortex_add_task(request):
        try:
            data = await request.json()
            task = data.get("task", "").strip()
            priority = data.get("priority", 0)
            
            if not task:
                return web.json_response({"error": "Empty task"}, status=400)
            
            await asyncio.shield(cortex.add_task(task, priority))
            logger.info(f"Task added: {task} with priority {priority}")
            return web.json_response({"status": "Task added successfully"})
            
        except Exception as e:
            logger.error(f"Error in cortex_add_task: {str(e)}")
            raise

    async def hippocampus_retrieve_memory(request):
        try:
            data = await request.json()
            query = data.get("query", "").strip()
            
            if not query:
                return web.json_response({"error": "Empty query"}, status=400)
            
            response = await asyncio.shield(hippocampus.retrieve_memory(query))
            logger.info(f"Memory retrieved for query '{query}': {response}")
            return web.json_response({"response": response})
            
        except Exception as e:
            logger.error(f"Error in hippocampus_retrieve_memory: {str(e)}")
            raise

    async def list_memories(request):
        try:
            memories = await asyncio.shield(hippocampus.list_all_memories())
            logger.info(f"All memories listed: {memories}")
            return web.json_response({"memories": memories})
        except Exception as e:
            logger.error(f"Error in list_memories: {str(e)}")
            raise

    async def server_status(request):
        logger.info("Server status requested")
        return web.json_response({"status": f"Server is running on port {PORT}"})

    async def upload_document(request):
        try:
            reader = await request.multipart()
            field = await reader.next()
            
            if not field or not field.filename:
                return web.json_response({"error": "No file provided"}, status=400)
            
            filename = field.filename
            file_path = f"uploads/{filename}"
            
            with open(file_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            
            documents = await asyncio.shield(
                document_processor.process_uploaded_document(file_path)
            )
            logger.info(f"Document uploaded and processed: {filename}")
            return web.json_response({
                "status": "Document processed successfully",
                "documents": documents
            })
            
        except Exception as e:
            logger.error(f"Error in upload_document: {str(e)}")
            raise

    # Configuration de l'application
    app = web.Application(middlewares=[error_middleware])
    app.router.add_post('/bouche/respond', bouche_respond)
    app.router.add_post('/cortex/add_task', cortex_add_task)
    app.router.add_post('/hippocampus/retrieve_memory', hippocampus_retrieve_memory)
    app.router.add_get('/hippocampus/memories', list_memories)
    app.router.add_get('/status', server_status)
    app.router.add_post('/documents/upload', upload_document)

    return app

if __name__ == "__main__":
    try:
        app = asyncio.run(start_server())
        web.run_app(app, host='localhost', port=PORT)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
