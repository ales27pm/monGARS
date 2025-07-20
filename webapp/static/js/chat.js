const { fastapiUrl, userId } = window.chatConfig || {};
if (!fastapiUrl || !userId) {
  alert('Configuration missing.');
  throw new Error('Missing fastapiUrl or userId');
}
let socket;
function addMessage(message) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message";
  msgDiv.textContent = `Query: ${message.query} | Response: ${message.response} | ${new Date(message.timestamp).toLocaleString('fr-CA')}`;
  chatBox.appendChild(msgDiv);
  chatBox.scroll({ top: chatBox.scrollHeight, behavior: 'smooth' });
}
function showError(message) {
  const errorAlert = document.getElementById("error-alert");
  document.getElementById("error-message").textContent = message;
  errorAlert.classList.remove("d-none");
}
let failureCount = 0;
function connectWebSocket() {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProtocol}://${window.location.host}/ws/chat/?user_id=${userId}`;
  socket = new WebSocket(wsUrl);
  socket.onopen = () => {
    document.getElementById("status-indicator").textContent = "Connecté";
    failureCount = 0;
    console.log("WebSocket connected");
  };
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data && data.query && data.response) {
      addMessage(data);
    }
  };
  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    showError("Erreur de connexion WebSocket");
  };
  socket.onclose = (event) => {
    failureCount += 1;
    const delay = Math.min(30000, 1000 * 2 ** failureCount);
    if (failureCount > 5) {
      document.getElementById("status-indicator").textContent = "Connexion échouée";
      console.error("WebSocket closed too many times:", event);
      return;
    }
    document.getElementById("status-indicator").textContent = `Déconnecté. Reconnexion dans ${delay / 1000}s...`;
    console.warn("WebSocket closed:", event);
    setTimeout(connectWebSocket, delay);
  };
}
connectWebSocket();
document.getElementById("toggle-dark-mode").addEventListener("click", () => {
  document.body.classList.toggle("dark-mode");
  const btn = document.getElementById("toggle-dark-mode");
  btn.textContent = document.body.classList.contains("dark-mode") ? "Mode Clair" : "Mode Sombre";
});
document.getElementById("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("message-input");
  const message = input.value.trim();
  if (!message) return;
  try {
    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("query", message);
    formData.append("session_id", "session1");
    const response = await fetch(`${fastapiUrl}/api/v1/conversation/chat`, {
      method: "POST",
      body: formData
    });
    if (!response.ok) throw new Error("Erreur lors de l'envoi du message");
    input.value = "";
  } catch (error) {
    showError(error.message);
  }
});
