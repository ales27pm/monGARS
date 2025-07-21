const { fastapiUrl, userId } = window.chatConfig || {};
if (!fastapiUrl || !userId) {
  alert('Configuration missing.');
  throw new Error('Missing fastapiUrl or userId');
}
const historyElement = document.getElementById('chat-history');
let chatHistory = [];
const seenMessages = new Set();
if (historyElement) {
  try {
    chatHistory = JSON.parse(historyElement.textContent);
    if (chatHistory.error) {
      showError(chatHistory.error);
      chatHistory = [];
    }
  } catch (e) {
    console.error('Failed to parse history', e);
  }
  historyElement.remove();
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

function addMessageIfUnique(message) {
  const key = message.id ?? message.timestamp ?? `${message.query}|${message.response}`;
  if (seenMessages.has(key)) {
    return;
  }
  seenMessages.add(key);
  addMessage(message);
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
      addMessageIfUnique(data);
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
chatHistory.forEach(addMessageIfUnique);
const darkModeKey = 'dark-mode';
const toggleBtn = document.getElementById("toggle-dark-mode");
function applyDarkMode(enabled) {
  document.body.classList.toggle("dark-mode", enabled);
  if (toggleBtn) {
    toggleBtn.textContent = enabled ? "Mode Clair" : "Mode Sombre";
  }
}
applyDarkMode(localStorage.getItem(darkModeKey) === '1');
if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    const enabled = !document.body.classList.contains("dark-mode");
    applyDarkMode(enabled);
    localStorage.setItem(darkModeKey, enabled ? '1' : '0');
  });
}
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
    const data = await response.json();
    addMessageIfUnique({
      query: message,
      response: data.response,
      timestamp: Date.now()
    });
    input.value = "";
  } catch (error) {
    showError(error.message);
  }
});
