import React, { useState } from 'react';
import {
  Alert,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Diagnostics from '../native/diagnostics';
import { authenticate } from '../services/authService';
import { settings } from '../services/config';
import { useChatStore } from '../store/chatStore';

const SettingsScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const { session, connection, setSession, logout } = useChatStore();

  const handleLogin = async () => {
    setBusy(true);
    try {
      const auth = await authenticate({ username, password });
      await setSession({
        username: auth.username,
        token: auth.accessToken,
      });
      setPassword('');
      Alert.alert('Connexion reussie', `Session active pour ${auth.username}.`);
    } catch (error) {
      console.error('[Settings] login failed', error);
      Alert.alert('Erreur', 'Identifiants invalides ou serveur indisponible.');
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    Alert.alert('Session fermee', 'Le jeton local a ete supprime.');
  };

  const enableDiagnostics = async () => {
    try {
      await Diagnostics.enablePacketTunnel({ serverAddress: '10.0.0.1' });
      Alert.alert('Tunnel active', 'Le tunnel de diagnostic est pret.');
    } catch (error) {
      Alert.alert('Erreur', (error as Error).message);
    }
  };

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Authentification</Text>
        <Text style={styles.description}>
          Cette application native utilise le meme JWT que le webapp Django,
          mais se connecte directement aux endpoints FastAPI.
        </Text>
        <TextInput
          accessibilityLabel="Nom d utilisateur"
          placeholder="Nom d utilisateur"
          placeholderTextColor="#64748b"
          autoCapitalize="none"
          style={styles.input}
          value={username}
          onChangeText={setUsername}
        />
        <TextInput
          accessibilityLabel="Mot de passe"
          placeholder="Mot de passe"
          placeholderTextColor="#64748b"
          secureTextEntry
          style={styles.input}
          value={password}
          onChangeText={setPassword}
        />
        <Pressable
          style={[styles.primaryButton, busy && styles.buttonDisabled]}
          disabled={busy}
          onPress={handleLogin}
        >
          <Text style={styles.primaryButtonText}>
            {busy ? 'Connexion…' : 'Se connecter'}
          </Text>
        </Pressable>
        {session ? (
          <View style={styles.sessionCard}>
            <Text style={styles.sessionTitle}>Session active</Text>
            <Text style={styles.sessionText}>
              Utilisateur: {session.username}
            </Text>
            <Text style={styles.sessionText}>
              Etat temps reel: {connection.status}
            </Text>
            <Text style={styles.sessionToken}>
              JWT: {session.token.slice(0, 18)}…
            </Text>
            <Pressable style={styles.secondaryButton} onPress={handleLogout}>
              <Text style={styles.secondaryButtonText}>Se deconnecter</Text>
            </Pressable>
          </View>
        ) : null}
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Configuration</Text>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>Base URL</Text>
          <Text style={styles.configValue}>{settings.baseUrl}</Text>
        </View>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>API</Text>
          <Text style={styles.configValue}>{settings.apiBaseUrl}</Text>
        </View>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>WebSocket</Text>
          <Text style={styles.configValue}>{settings.websocketUrl}</Text>
        </View>
        <View style={styles.configRow}>
          <Text style={styles.configLabel}>Embedding</Text>
          <Text style={styles.configValue}>
            {settings.embedServiceUrl ?? 'Non configure'}
          </Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Diagnostics natifs</Text>
        <Text style={styles.description}>
          Les hooks existants pour la capture reseau restent disponibles dans le
          client React Native.
        </Text>
        <Pressable style={styles.secondaryButton} onPress={enableDiagnostics}>
          <Text style={styles.secondaryButtonText}>
            Activer le tunnel de diagnostic
          </Text>
        </Pressable>
        <Text style={styles.footerText}>Plateforme: {Platform.OS}</Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#060b16',
  },
  container: {
    padding: 20,
    gap: 18,
  },
  card: {
    borderRadius: 28,
    padding: 20,
    backgroundColor: '#0d1525',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 14,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#f8fafc',
  },
  description: {
    color: '#94a3b8',
    lineHeight: 21,
  },
  input: {
    borderWidth: 1,
    borderColor: '#23314d',
    borderRadius: 18,
    padding: 14,
    backgroundColor: '#08111f',
    color: '#f8fafc',
  },
  primaryButton: {
    backgroundColor: '#f97316',
    paddingVertical: 14,
    borderRadius: 18,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#fff7ed',
    fontWeight: '800',
  },
  secondaryButton: {
    backgroundColor: '#172033',
    paddingVertical: 14,
    borderRadius: 18,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#dbeafe',
    fontWeight: '700',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  sessionCard: {
    borderRadius: 20,
    padding: 16,
    backgroundColor: '#07101d',
    gap: 8,
  },
  sessionTitle: {
    color: '#fbbf24',
    fontWeight: '800',
  },
  sessionText: {
    color: '#e2e8f0',
  },
  sessionToken: {
    color: '#93c5fd',
    fontFamily: 'Courier',
  },
  configRow: {
    gap: 6,
  },
  configLabel: {
    color: '#64748b',
    fontSize: 12,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  configValue: {
    color: '#f8fafc',
  },
  footerText: {
    color: '#64748b',
  },
});

export default SettingsScreen;
