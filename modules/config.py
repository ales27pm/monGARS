import logging

# Database Configuration
DB_CONFIG = {
    "dbname": "hippocampus",
    "user": "app_user",
    "password": "1406",
    "host": "localhost",
    "port": 5432
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "logs/mongars.log"
}

# LLM Configuration
LLM_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "dolphin-mistral:7b-v2.8-q5_1"
}
