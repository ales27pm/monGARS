import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Linking,
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
import { useInferenceStore } from '../store/inferenceStore';

function formatBytes(bytes: number): string {
  if (bytes <= 0) {
    return 'Non installe';
  }
  return `${(bytes / 1_000_000_000).toFixed(2)} Go`;
}

const SettingsScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [modelBusy, setModelBusy] = useState(false);
  const {
    session,
    connection,
    loading: chatRequestInProgress,
    setSession,
    setInferenceBackend,
    logout,
  } = useChatStore();
  const {
    backend,
    status: modelStatus,
    progress: modelProgress,
    error: modelError,
    activeRequestId,
    initialize: initializeInference,
    prepareModel,
    deleteModel,
  } = useInferenceStore();

  useEffect(() => {
    initializeInference().catch((error) => {
      console.warn('[Settings] Core ML status failed', error);
    });
  }, [initializeInference]);

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

  const selectBackend = async (nextBackend: 'server' | 'on-device') => {
    if (nextBackend === backend) {
      return;
    }
    try {
      await setInferenceBackend(nextBackend);
    } catch (error) {
      Alert.alert('Backend indisponible', (error as Error).message);
    }
  };

  const downloadModel = async () => {
    setModelBusy(true);
    try {
      await prepareModel();
      Alert.alert(
        'Modele pret',
        'Le modele Core ML est verifie et charge sur l iPhone.',
      );
    } catch (error) {
      Alert.alert('Modele indisponible', (error as Error).message);
    } finally {
      setModelBusy(false);
    }
  };

  const confirmDownload = () => {
    Alert.alert(
      'Telecharger le modele local?',
      'Le telechargement fait environ 1,57 Go et exige au moins 2,5 Go libres. Utilisez de preference le Wi-Fi.',
      [
        { text: 'Annuler', style: 'cancel' },
        { text: 'Telecharger', onPress: downloadModel },
      ],
    );
  };

  const removeModel = async () => {
    setModelBusy(true);
    try {
      await deleteModel();
      Alert.alert('Modele supprime', 'Les poids locaux ont ete effaces.');
    } catch (error) {
      Alert.alert('Suppression impossible', (error as Error).message);
    } finally {
      setModelBusy(false);
    }
  };

  const confirmRemoveModel = () => {
    Alert.alert(
      'Supprimer le modele local?',
      'Une nouvelle inference locale exigera de le telecharger de nouveau.',
      [
        { text: 'Annuler', style: 'cancel' },
        { text: 'Supprimer', style: 'destructive', onPress: removeModel },
      ],
    );
  };

  const modelInstalled = modelStatus.installedBytes > 0;
  const generationInProgress =
    chatRequestInProgress ||
    activeRequestId !== null ||
    modelStatus.phase === 'generating';
  const onDeviceBackendUnavailable =
    Platform.OS !== 'ios' || modelStatus.phase === 'unavailable';
  const modelOperationInProgress =
    modelBusy ||
    activeRequestId !== null ||
    ['downloading', 'verifying', 'compiling', 'loading'].includes(
      modelStatus.phase,
    );

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Moteur d inference</Text>
        <Text style={styles.description}>
          Choisissez explicitement le serveur ou Core ML. Aucun basculement
          silencieux vers le reseau n est effectue.
        </Text>
        <View style={styles.backendRow}>
          <Pressable
            accessibilityRole="button"
            disabled={generationInProgress}
            onPress={() => selectBackend('server')}
            style={[
              styles.backendButton,
              backend === 'server' && styles.backendButtonActive,
              generationInProgress && styles.buttonDisabled,
            ]}
          >
            <Text
              style={[
                styles.backendButtonText,
                backend === 'server' && styles.backendButtonTextActive,
              ]}
            >
              Serveur
            </Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            disabled={
              onDeviceBackendUnavailable ||
              modelOperationInProgress ||
              generationInProgress
            }
            onPress={() => selectBackend('on-device')}
            style={[
              styles.backendButton,
              backend === 'on-device' && styles.backendButtonActive,
              (onDeviceBackendUnavailable ||
                modelOperationInProgress ||
                generationInProgress) &&
                styles.buttonDisabled,
            ]}
          >
            <Text
              style={[
                styles.backendButtonText,
                backend === 'on-device' && styles.backendButtonTextActive,
              ]}
            >
              Sur l iPhone
            </Text>
          </Pressable>
        </View>

        <View style={styles.modelCard}>
          <View style={styles.modelHeader}>
            <View style={styles.modelHeaderText}>
              <Text style={styles.modelName}>
                {modelStatus.displayName ??
                  'Dolphin 3.0 · Llama 3.2 3B · Core ML INT4'}
              </Text>
              <Text style={styles.modelIdentifier}>
                {modelStatus.modelId ?? 'ales27pm/Dolphin3.0-CoreML'}
              </Text>
            </View>
            <Text style={styles.modelPhase}>{modelStatus.phase}</Text>
          </View>
          <View style={styles.modelFacts}>
            <Text style={styles.modelFact}>
              Disque: {formatBytes(modelStatus.installedBytes)}
            </Text>
            <Text style={styles.modelFact}>
              Contexte: {modelStatus.contextLength || 2048} jetons
            </Text>
            <Text style={styles.modelFact}>
              Minimum: iOS {modelStatus.minimumIOSVersion || 18}
            </Text>
          </View>

          {modelProgress ? (
            <View style={styles.progressContainer}>
              <View style={styles.progressTrack}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${Math.round(
                        modelProgress.fractionCompleted * 100,
                      )}%`,
                    },
                  ]}
                />
              </View>
              <Text style={styles.progressText}>
                {modelProgress.detail ?? 'Preparation'} ·{' '}
                {Math.round(modelProgress.fractionCompleted * 100)}%
              </Text>
            </View>
          ) : null}

          {modelError || modelStatus.detail ? (
            <Text style={styles.modelDetail}>
              {modelError ?? modelStatus.detail}
            </Text>
          ) : null}

          <View style={styles.modelActions}>
            {modelOperationInProgress ? (
              <ActivityIndicator color="#f59e0b" />
            ) : modelInstalled ? (
              <Pressable
                style={styles.destructiveButton}
                onPress={confirmRemoveModel}
              >
                <Text style={styles.destructiveButtonText}>
                  Supprimer les poids
                </Text>
              </Pressable>
            ) : (
              <Pressable
                disabled={modelStatus.phase === 'unavailable'}
                style={[
                  styles.primaryButton,
                  modelStatus.phase === 'unavailable' && styles.buttonDisabled,
                ]}
                onPress={confirmDownload}
              >
                <Text style={styles.primaryButtonText}>
                  Telecharger et verifier
                </Text>
              </Pressable>
            )}
            <Pressable
              style={styles.linkButton}
              onPress={() =>
                Linking.openURL(
                  'https://huggingface.co/ales27pm/Dolphin3.0-CoreML/tree/main/Dolphin3.0-Llama3.2-3B-stateful-int4.mlpackage',
                )
              }
            >
              <Text style={styles.linkButtonText}>Voir la provenance HF</Text>
            </Pressable>
          </View>
          <Text style={styles.footerText}>
            Artefact stateful INT4 epingle et verifie par SHA-256. Source
            Dolphin/Llama 3.2 sous licence communautaire Llama; embeddings
            toujours cote serveur.
          </Text>
        </View>
      </View>

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
  backendRow: {
    flexDirection: 'row',
    gap: 10,
  },
  backendButton: {
    flex: 1,
    borderRadius: 999,
    paddingVertical: 12,
    alignItems: 'center',
    backgroundColor: '#172033',
    borderWidth: 1,
    borderColor: '#23314d',
  },
  backendButtonActive: {
    backgroundColor: '#f59e0b',
    borderColor: '#fbbf24',
  },
  backendButtonText: {
    color: '#cbd5e1',
    fontWeight: '700',
  },
  backendButtonTextActive: {
    color: '#111827',
  },
  modelCard: {
    borderRadius: 22,
    padding: 16,
    backgroundColor: '#07101d',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 12,
  },
  modelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 12,
  },
  modelHeaderText: {
    flex: 1,
    gap: 4,
  },
  modelName: {
    color: '#f8fafc',
    fontWeight: '800',
  },
  modelIdentifier: {
    color: '#94a3b8',
    fontSize: 11,
  },
  modelPhase: {
    color: '#fbbf24',
    fontSize: 11,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  modelFacts: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  modelFact: {
    color: '#cbd5e1',
    fontSize: 12,
  },
  progressContainer: {
    gap: 7,
  },
  progressTrack: {
    height: 8,
    borderRadius: 999,
    overflow: 'hidden',
    backgroundColor: '#1e293b',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#f59e0b',
  },
  progressText: {
    color: '#cbd5e1',
    fontSize: 12,
  },
  modelDetail: {
    color: '#fca5a5',
    lineHeight: 19,
  },
  modelActions: {
    gap: 10,
  },
  destructiveButton: {
    backgroundColor: '#7f1d1d',
    paddingVertical: 13,
    borderRadius: 16,
    alignItems: 'center',
  },
  destructiveButtonText: {
    color: '#fee2e2',
    fontWeight: '800',
  },
  linkButton: {
    paddingVertical: 8,
    alignItems: 'center',
  },
  linkButtonText: {
    color: '#93c5fd',
    fontWeight: '700',
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
