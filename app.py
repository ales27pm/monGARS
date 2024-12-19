import streamlit as st
import requests
import os
import logging
import subprocess
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

BACKEND_URL = "http://localhost:8081"

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    raise_on_status=False,
    raise_on_redirect=False
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

def check_backend_status():
    try:
        response = http.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            logging.info("Backend status: actif")
            return True
        else:
            logging.warning(f"Backend status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de connexion : {e}")
        return False

def start_backend():
    try:
        subprocess.run(["python", "main.py"], check=True)
        logging.info("Backend started successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start backend: {e}")
        st.error("Failed to start backend. Please check the logs for more details.")

st.title("monGARS - Interface Web")

# Section : Vérification du statut du serveur
st.header("⚙️ Statut du serveur")
if st.button("Vérifier le statut du backend"):
    if check_backend_status():
        st.success("Le serveur backend est actif.")
    else:
        st.warning("Le serveur backend n'est pas actif. Tentative de démarrage du backend...")
        start_backend()
        if check_backend_status():
            st.success("Le serveur backend est maintenant actif.")
        else:
            st.error("Le serveur backend n'a pas pu être démarré.")

# Section 1 : Chat avec Bouche
st.header("💬 Chat avec Bouche")
query = st.text_input("Posez votre question :", key="query_input")
if st.button("Envoyer"):
    if check_backend_status():
        with st.spinner("Bouche réfléchit..."):
            try:
                response = http.post(f"{BACKEND_URL}/bouche/respond", json={"query": query}, timeout=10)
                st.success("Réponse de Bouche :")
                st.write(response.json().get("response", "Aucune réponse trouvée."))
                logging.info(f"Query: {query}, Response: {response.json().get('response', 'Aucune réponse trouvée.')}")
            except requests.exceptions.Timeout:
                st.error("La requête a pris trop de temps. Veuillez réessayer plus tard.")
                logging.error("La requête a pris trop de temps.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion : {e}")
                logging.error(f"Erreur de connexion : {e}")
            except Exception as e:
                st.error(f"Erreur : {e}")
                logging.error(f"Erreur : {e}")
    else:
        st.warning("Le serveur backend n'est pas actif. Veuillez vérifier le statut du serveur.")

# Section 2 : Récupérer une mémoire depuis Hippocampus
st.header("🧠 Récupérer une mémoire (Hippocampus)")
memory_query = st.text_input("Rechercher dans la mémoire :", key="memory_query")
if st.button("Récupérer Mémoire"):
    if check_backend_status():
        with st.spinner("Recherche dans Hippocampus..."):
            try:
                response = http.post(f"{BACKEND_URL}/hippocampus/retrieve_memory", json={"query": memory_query})
                st.success("Mémoire trouvée :")
                st.write(response.json().get("response", "Aucune mémoire correspondante."))
                logging.info(f"Memory query: {memory_query}, Response: {response.json().get('response', 'Aucune mémoire correspondante.')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion : {e}")
                logging.error(f"Erreur de connexion : {e}")
            except Exception as e:
                st.error(f"Erreur : {e}")
                logging.error(f"Erreur : {e}")
    else:
        st.warning("Le serveur backend n'est pas actif. Veuillez vérifier le statut du serveur.")

# Section 3 : Ajouter une tâche dans Cortex
st.header("📝 Ajouter une tâche (Cortex)")
task_name = st.text_input("Nom de la tâche :", key="task_name")
priority = st.slider("Priorité de la tâche :", 1, 10, 5, key="task_priority")
if st.button("Ajouter Tâche"):
    if check_backend_status():
        with st.spinner("Ajout de la tâche dans Cortex..."):
            try:
                response = http.post(f"{BACKEND_URL}/cortex/add_task", json={"task": task_name, "priority": priority})
                st.success(response.json().get("status", "Tâche ajoutée avec succès."))
                logging.info(f"Task: {task_name}, Priority: {priority}, Status: {response.json().get('status', 'Tâche ajoutée avec succès.')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion : {e}")
                logging.error(f"Erreur de connexion : {e}")
            except Exception as e:
                st.error(f"Erreur : {e}")
                logging.error(f"Erreur : {e}")
    else:
        st.warning("Le serveur backend n'est pas actif. Veuillez vérifier le statut du serveur.")

# Section 4 : Gestion des documents
st.header("📂 Upload et traitement de documents")
uploaded_file = st.file_uploader("Choisissez un fichier (texte, PDF ou Word)", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    if check_backend_status():
        with st.spinner("Traitement du fichier..."):
            try:
                # Sauvegarde temporaire
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Envoi à l'API pour traitement
                with open(file_path, "rb") as f:
                    response = http.post(f"{BACKEND_URL}/documents/upload", files={"file": f})
                
                st.success("Fichier traité avec succès !")
                st.write(response.text)
                logging.info(f"Uploaded file: {uploaded_file.name}, Response: {response.text}")

                # Aperçu du contenu extrait
                st.subheader("Aperçu des données extraites :")
                if uploaded_file.name.endswith(".txt"):
                    st.text(uploaded_file.getvalue().decode("utf-8")[:500])  # 500 premiers caractères
                elif uploaded_file.name.endswith(".pdf"):
                    st.text("Le contenu extrait du PDF sera affiché ici après traitement.")
                elif uploaded_file.name.endswith(".docx"):
                    st.text("Le contenu extrait du document Word sera affiché ici après traitement.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion : {e}")
                logging.error(f"Erreur de connexion : {e}")
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier : {e}")
                logging.error(f"Erreur lors du traitement du fichier : {e}")
    else:
        st.warning("Le serveur backend n'est pas actif. Veuillez vérifier le statut du serveur.")
