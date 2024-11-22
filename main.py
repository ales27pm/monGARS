
import uvicorn
from app.api.routes import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=4, http="h11", log_level="info", compression="gzip")
