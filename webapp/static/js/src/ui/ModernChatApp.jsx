import React, { useMemo, useState } from "react";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";

import { resolveConfig } from "../config.js";
import { createHttpService } from "../services/http.js";
import { createAuthService } from "../services/auth.js";

const QUICK_PRESETS = {
  code: "Je souhaite écrire du code…",
  summarize: "Résume la dernière conversation.",
  explain: "Explique ta dernière réponse plus simplement.",
};

export function ModernChatApp() {
  const config = useMemo(() => resolveConfig(globalThis.chatConfig || {}), []);
  const http = useMemo(
    () => createHttpService({ config, auth: createAuthService(config) }),
    [config],
  );
  const [messages, setMessages] = useState([]);
  const [value, setValue] = useState("");
  const [mode, setMode] = useState("chat");
  const [status, setStatus] = useState("Prêt à discuter.");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [snackbar, setSnackbar] = useState(false);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: "light",
          primary: {
            main: "#0d6efd",
          },
          secondary: {
            main: "#6c757d",
          },
        },
        typography: {
          fontFamily: "Inter, Arial, Helvetica, sans-serif",
        },
      }),
    [],
  );

  async function sendMessage(text) {
    if (!text?.trim()) {
      setStatus("Entrez un message avant d'envoyer.");
      return;
    }

    const trimmed = text.trim();
    const userEntry = { id: Date.now() + "-user", role: "user", text: trimmed };
    setMessages((prev) => [...prev, userEntry]);
    setValue("");
    setStatus(
      mode === "embed" ? "Génération d'embeddings…" : "En attente de réponse…",
    );
    setLoading(true);

    try {
      if (mode === "embed") {
        const response = await http.postEmbed(trimmed);
        const assistantEntry = {
          id: Date.now() + "-assistant",
          role: "assistant",
          text: `Embedding généré: ${response.vectors?.length || 0} vecteur(s).`,
        };
        setMessages((prev) => [...prev, assistantEntry]);
        setStatus("Embedding terminé avec succès.");
      } else {
        const response = await http.postChat(trimmed);
        const assistantEntry = {
          id: Date.now() + "-assistant",
          role: "assistant",
          text: response?.response || "Aucune réponse du serveur.",
        };
        setMessages((prev) => [...prev, assistantEntry]);
        setStatus("Réponse reçue.");
      }
      setError(null);
    } catch (err) {
      const message = (err?.message) || "Erreur lors de l'envoi.";
      setError(message);
      setStatus("Erreur lors de l'opération.");
      setSnackbar(true);
    } finally {
      setLoading(false);
    }
  }

  const handleQuickAction = (action) => {
    const preset = QUICK_PRESETS[action] || action;
    setValue(preset);
    sendMessage(preset);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" color="primary" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            monGARS Chat Moderne
          </Typography>
          <Typography variant="caption" noWrap>
            {status}
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ mt: 3, mb: 3 }}>
        <Paper elevation={2} sx={{ p: 2, minHeight: "72vh" }}>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              mb: 1,
              alignItems: "center",
            }}
          >
            <Typography variant="subtitle1">Historique</Typography>
            <ToggleButtonGroup
              value={mode}
              size="small"
              exclusive
              onChange={(_, next) => {
                if (next) setMode(next);
              }}
              aria-label="Mode du chat"
            >
              <ToggleButton value="chat" aria-label="mode chat">
                Chat
              </ToggleButton>
              <ToggleButton value="embed" aria-label="mode embedding">
                Embedding
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          <Box
            sx={{
              maxHeight: "45vh",
              overflowY: "auto",
              mb: 2,
              borderRadius: 1,
              border: "1px solid #e0e0e0",
            }}
          >
            <List disablePadding>
              {messages.length === 0 ? (
                <ListItem>
                  <ListItemText primary="Aucune conversation pour le moment." />
                </ListItem>
              ) : (
                messages.map((msg) => (
                  <ListItem
                    key={msg.id}
                    sx={{
                      bgcolor:
                        msg.role === "user"
                          ? "rgba(13,110,253,0.08)"
                          : "rgba(117,116,255,0.08)",
                      mb: 1,
                      borderRadius: 1,
                      alignItems: "flex-start",
                    }}
                  >
                    <ListItemText
                      primary={msg.role === "user" ? "Vous" : "Assistant"}
                      secondary={msg.text}
                    />
                  </ListItem>
                ))
              )}
            </List>
          </Box>

          <Box
            sx={{
              display: "flex",
              gap: 1,
              flexDirection: { xs: "column", sm: "row" },
            }}
          >
            <TextField
              fullWidth
              variant="outlined"
              multiline
              minRows={2}
              maxRows={6}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              placeholder="Tapez votre message ici..."
              disabled={loading}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage(value);
                }
              }}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={() => sendMessage(value)}
              disabled={loading}
              sx={{ minWidth: 120 }}
            >
              {loading ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                "Envoyer"
              )}
            </Button>
          </Box>

          <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
            {Object.keys(QUICK_PRESETS).map((action) => (
              <Button
                key={action}
                variant="outlined"
                size="small"
                onClick={() => handleQuickAction(action)}
              >
                {action}
              </Button>
            ))}
          </Box>
        </Paper>
      </Container>

      <Snackbar
        open={snackbar}
        autoHideDuration={6000}
        onClose={() => setSnackbar(false)}
      >
        <Alert severity="error" onClose={() => setSnackbar(false)}>
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}
