
# Autonomous Assistant Project

## Overview
This project provides an AI-powered assistant with the following features:
- **Memory Management:** Add, retrieve, and connect memories dynamically.
- **Web Scraping:** Supports basic and dynamic content scraping (requires `playwright`).
- **Self-Awareness:** Tracks file dependencies, logs activities, and optimizes code.
- **Visualization:** Graph visualization of memory relationships.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Set Up Playwright (Optional)
For dynamic web scraping, install Playwright and its browsers:
```bash
pip install playwright
playwright install
```

### 3. Run the Backend
Start the FastAPI server:
```bash
python main.py
```

### 4. Run the Frontend
Serve the `static/` directory using Python:
```bash
cd static
python -m http.server 8001
```

### 5. Access the Application
- Backend API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Frontend: [http://127.0.0.1:8001](http://127.0.0.1:8001)

## Testing
To run the test suite:
```bash
pytest tests/
```

## Troubleshooting
- If `playwright` tests fail, ensure it is installed and browsers are set up.
