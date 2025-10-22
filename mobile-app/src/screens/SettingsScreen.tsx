import React, { useState } from 'react';
import {
  Alert,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
} from 'react-native';
import { authenticate } from '../services/authService';
import { useChatStore } from '../store/chatStore';
import Diagnostics from '../native/diagnostics';

const SettingsScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const { setToken, token } = useChatStore();

  const handleLogin = async () => {
    setBusy(true);
    try {
      const auth = await authenticate({ username, password });
      setToken(auth.accessToken);
      Alert.alert('Connexion réussie', `Expiration ${auth.expiresAt}`);
    } catch (error) {
      console.error('[Settings] login failed', error);
      Alert.alert('Erreur', 'Identifiants invalides');
    } finally {
      setBusy(false);
    }
  };

  const openEntitlementsDoc = () => {
    Linking.openURL(
      'https://developer.apple.com/documentation/bundleresources/entitlements',
    );
  };

  const enableDiagnostics = async () => {
    try {
      await Diagnostics.enablePacketTunnel({ serverAddress: '10.0.0.1' });
      Alert.alert('Tunnel activé', 'Le tunnel de diagnostic est actif.');
    } catch (error) {
      Alert.alert('Erreur', (error as Error).message);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.sectionTitle}>Authentification</Text>
      <TextInput
        accessibilityLabel="Nom d'utilisateur"
        placeholder="Nom d'utilisateur"
        autoCapitalize="none"
        style={styles.input}
        value={username}
        onChangeText={setUsername}
      />
      <TextInput
        accessibilityLabel="Mot de passe"
        placeholder="Mot de passe"
        secureTextEntry
        style={styles.input}
        value={password}
        onChangeText={setPassword}
      />
      <Pressable
        style={[styles.button, busy && styles.buttonDisabled]}
        disabled={busy}
        onPress={handleLogin}
      >
        <Text style={styles.buttonText}>
          {busy ? 'Connexion…' : 'Se connecter'}
        </Text>
      </Pressable>
      {token ? (
        <Text style={styles.token}>Token actif: {token.slice(0, 12)}…</Text>
      ) : null}

      <Text style={styles.sectionTitle}>Fonctionnalités iOS</Text>
      <Text style={styles.description}>
        Activez la reconnaissance vocale, le partage natif et les captures
        réseau via Network Extension.
      </Text>
      <Pressable style={styles.button} onPress={enableDiagnostics}>
        <Text style={styles.buttonText}>Activer le tunnel de diagnostic</Text>
      </Pressable>
      {Platform.OS === 'ios' ? (
        <Pressable style={styles.linkButton} onPress={openEntitlementsDoc}>
          <Text style={styles.linkText}>Guide des entitlements Apple</Text>
        </Pressable>
      ) : null}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 24,
    gap: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#0f172a',
  },
  input: {
    borderWidth: 1,
    borderColor: '#1e293b',
    borderRadius: 12,
    padding: 12,
    backgroundColor: '#f8fafc',
  },
  button: {
    backgroundColor: '#0f172a',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#f8fafc',
    fontWeight: '600',
  },
  description: {
    color: '#334155',
  },
  linkButton: {
    paddingVertical: 12,
  },
  linkText: {
    color: '#2563eb',
  },
  token: {
    color: '#1d4ed8',
    fontFamily: 'Courier',
  },
});

export default SettingsScreen;
