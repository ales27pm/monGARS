import subprocess
import sys
import time
import requests
import logging

BACKEND_URL = "http://localhost:8081"

# Configure logging
logging.basicConfig(
    filename="logs/start.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

def start_backend():
    logging.info("Démarrage du backend...")
    try:
        backend_process = subprocess.Popen([sys.executable, "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(20)  # Increased time to 20 seconds
        if check_backend_status():
            logging.info("Backend lancé avec succès.")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate(timeout=20)
            logging.info(f"Backend stdout: {stdout.decode()}")
            logging.error(f"Backend stderr: {stderr.decode()}")
            logging.error("Échec du démarrage du backend.")
            backend_process.terminate()
            sys.exit(1)
    except subprocess.TimeoutExpired:
        logging.error("Backend startup timed out.")
        backend_process.terminate()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du backend : {e}")
        sys.exit(1)

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de connexion : {e}")
        return False

def start_frontend():
    logging.info("Démarrage du frontend (Streamlit)...")
    try:
        frontend_process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Frontend lancé avec succès.")
        return frontend_process
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du frontend : {e}")
        sys.exit(1)

def handle_document_upload():
    logging.info("Gestion du processus de téléchargement de document...")

def main():
    try:
        # Lancer le backend
        backend = start_backend()
        
        # Lancer le frontend
        frontend = start_frontend()
        
        # Attendre que l'utilisateur arrête le script
        logging.info("Appuyez sur Ctrl+C pour arrêter les deux processus.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Arrêt en cours...")
        backend.terminate()
        frontend.terminate()
        logging.info("Tous les processus ont été arrêtés.")
    except Exception as e:
        logging.error(f"Erreur inattendue : {e}")
        if backend:
            backend.terminate()
        if frontend:
            frontend.terminate()

if __name__ == "__main__":
    main()