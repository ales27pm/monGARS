import React, { useEffect, useRef, useState } from 'react';
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
import {
  configureNativeOutlookClientId,
  connectNativeOutlook,
  disconnectNativeOutlook,
  getNativeOutlookConnectionStatus,
  nativeOutlookModuleAvailable,
  type NativeOutlookConnectionStatus,
} from '../native/outlook';
import { authenticate } from '../services/authService';
import { settings } from '../services/config';
import { getLocalConversationOwner, useChatStore } from '../store/chatStore';
import { useInferenceStore } from '../store/inferenceStore';

const MODEL_DOWNLOAD_BYTES = 1_825_812_981;
const MODEL_REQUIRED_FREE_DISK_BYTES = 5_000_000_000;

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
  const [outlookBusy, setOutlookBusy] = useState(false);
  const [outlookStatus, setOutlookStatus] =
    useState<NativeOutlookConnectionStatus | null>(null);
  const [outlookClientId, setOutlookClientId] = useState('');
  const [outlookError, setOutlookError] = useState<string | null>(null);
  const outlookRequestVersion = useRef(0);
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
  const outlookOwnerId = session ? getLocalConversationOwner(session) : null;

  useEffect(() => {
    initializeInference().catch((error) => {
      console.warn('[Settings] Core ML status failed', error);
    });
  }, [initializeInference]);

  useEffect(() => {
    const requestVersion = ++outlookRequestVersion.current;
    setOutlookStatus(null);
    setOutlookClientId('');
    setOutlookError(null);
    setOutlookBusy(false);
    if (!nativeOutlookModuleAvailable || !outlookOwnerId) {
      return;
    }
    getNativeOutlookConnectionStatus(outlookOwnerId)
      .then((status) => {
        if (outlookRequestVersion.current === requestVersion) {
          setOutlookStatus(status);
          setOutlookError(null);
        }
      })
      .catch((error) => {
        if (outlookRequestVersion.current === requestVersion) {
          setOutlookError(
            error instanceof Error
              ? error.message
              : 'Statut Outlook indisponible.',
          );
        }
      });
    return () => {
      if (outlookRequestVersion.current === requestVersion) {
        outlookRequestVersion.current += 1;
      }
    };
  }, [outlookOwnerId]);

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
    Alert.alert('Session fermee', 'Le jeton en memoire a ete supprime.');
  };

  const connectOutlook = async () => {
    if (!outlookOwnerId) {
      setOutlookError('Connectez-vous à monGARS avant de connecter Outlook.');
      return;
    }
    const ownerId = outlookOwnerId;
    const requestVersion = ++outlookRequestVersion.current;
    setOutlookBusy(true);
    setOutlookError(null);
    try {
      const status = await connectNativeOutlook(ownerId);
      if (
        outlookRequestVersion.current !== requestVersion ||
        getLocalConversationOwner(useChatStore.getState().session) !== ownerId
      ) {
        return;
      }
      setOutlookStatus(status);
      Alert.alert(
        'Outlook connecté',
        `Compte actif: ${status.account ?? 'Microsoft'}.`,
      );
    } catch (error) {
      if (outlookRequestVersion.current !== requestVersion) {
        return;
      }
      const message =
        error instanceof Error
          ? error.message
          : 'Connexion Outlook impossible.';
      setOutlookError(message);
      Alert.alert('Connexion Outlook impossible', message);
    } finally {
      if (outlookRequestVersion.current === requestVersion) {
        setOutlookBusy(false);
      }
    }
  };

  const configureOutlook = async () => {
    if (!outlookOwnerId) {
      setOutlookError('Connectez-vous à monGARS avant de configurer Outlook.');
      return;
    }
    const ownerId = outlookOwnerId;
    const requestVersion = ++outlookRequestVersion.current;
    setOutlookBusy(true);
    setOutlookError(null);
    try {
      const status = await configureNativeOutlookClientId(
        ownerId,
        outlookClientId,
      );
      if (
        outlookRequestVersion.current !== requestVersion ||
        getLocalConversationOwner(useChatStore.getState().session) !== ownerId
      ) {
        return;
      }
      setOutlookStatus(status);
      setOutlookClientId('');
      Alert.alert(
        'Outlook configuré',
        "L'identifiant public Microsoft a été enregistré sur cet appareil.",
      );
    } catch (error) {
      if (outlookRequestVersion.current !== requestVersion) {
        return;
      }
      const message =
        error instanceof Error
          ? error.message
          : 'Configuration Outlook impossible.';
      setOutlookError(message);
      Alert.alert('Configuration Outlook impossible', message);
    } finally {
      if (outlookRequestVersion.current === requestVersion) {
        setOutlookBusy(false);
      }
    }
  };

  const disconnectOutlook = async () => {
    if (!outlookOwnerId) {
      setOutlookError('Connectez-vous à monGARS avant de déconnecter Outlook.');
      return;
    }
    const ownerId = outlookOwnerId;
    const requestVersion = ++outlookRequestVersion.current;
    setOutlookBusy(true);
    setOutlookError(null);
    try {
      const status = await disconnectNativeOutlook(ownerId);
      if (
        outlookRequestVersion.current !== requestVersion ||
        getLocalConversationOwner(useChatStore.getState().session) !== ownerId
      ) {
        return;
      }
      setOutlookStatus(status);
      Alert.alert(
        'Outlook déconnecté',
        'Les jetons Microsoft ont été supprimés du trousseau iOS.',
      );
    } catch (error) {
      if (outlookRequestVersion.current !== requestVersion) {
        return;
      }
      const message =
        error instanceof Error
          ? error.message
          : 'Déconnexion Outlook impossible.';
      setOutlookError(message);
      Alert.alert('Déconnexion Outlook impossible', message);
    } finally {
      if (outlookRequestVersion.current === requestVersion) {
        setOutlookBusy(false);
      }
    }
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
      `Le telechargement fait environ ${formatBytes(
        MODEL_DOWNLOAD_BYTES,
      )} et exige au moins ${formatBytes(
        MODEL_REQUIRED_FREE_DISK_BYTES,
      )} libres. Utilisez de preference le Wi-Fi.`,
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
                  'https://huggingface.co/ales27pm/Dolphin3.0-CoreML/tree/95671cf9a2f56d2a381816ae264cd9aae335d96f/Dolphin3.0-Llama3.2-3B-stateful-int4.mlpackage',
                )
              }
            >
              <Text style={styles.linkButtonText}>Voir la provenance HF</Text>
            </Pressable>
          </View>
          <Text style={styles.footerText}>
            Artefact stateful INT4 epingle et verifie par SHA-256. Source
            Dolphin/Llama 3.2 sous licence communautaire Llama. Built with
            Llama. Embeddings toujours cote serveur.
          </Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Authentification</Text>
        <Text style={styles.description}>
          Le jeton de session reste uniquement en memoire et doit etre obtenu de
          nouveau apres le redemarrage de l application.
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
        <Text style={styles.sectionTitle}>Outlook · Microsoft Graph</Text>
        <Text style={styles.description}>
          Connexion OAuth 2.0 avec PKCE, sans secret client. Les jetons ne sont
          jamais transmis à React Native et restent dans le trousseau iOS.
        </Text>
        {!session ? (
          <Text style={styles.warningText}>
            Connectez-vous à monGARS avant de connecter un compte Outlook.
          </Text>
        ) : !nativeOutlookModuleAvailable ? (
          <Text style={styles.warningText}>
            Module Outlook natif indisponible sur cette plateforme.
          </Text>
        ) : outlookStatus ? (
          <View style={styles.sessionCard}>
            <Text style={styles.sessionTitle}>
              {outlookStatus.configured ? 'Configuré' : 'Non configuré'} ·{' '}
              {outlookStatus.connected ? 'Connecté' : 'Non connecté'}
            </Text>
            {outlookStatus.account ? (
              <Text style={styles.sessionText}>
                Compte: {outlookStatus.account}
              </Text>
            ) : null}
            <Text selectable style={styles.outlookDetail}>
              Redirection: {outlookStatus.redirectUri}
            </Text>
            <Text style={styles.outlookDetail}>{outlookStatus.detail}</Text>
            <Text style={styles.outlookDetail}>
              Autorisations: {outlookStatus.requiredScopes.join(', ')}
            </Text>
            {!outlookStatus.configured ? (
              <View>
                <Text style={styles.outlookDetail}>
                  Saisissez l’ID d’application (client) public de votre
                  inscription Microsoft Entra. Aucun secret client n’est
                  accepté.
                </Text>
                <TextInput
                  accessibilityLabel="ID client Microsoft"
                  autoCapitalize="none"
                  autoCorrect={false}
                  editable={!outlookBusy}
                  placeholder="00000000-0000-0000-0000-000000000000"
                  placeholderTextColor="#64748b"
                  style={styles.input}
                  value={outlookClientId}
                  onChangeText={setOutlookClientId}
                />
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Enregistrer l’ID client Microsoft"
                  disabled={outlookBusy || !outlookClientId.trim()}
                  style={[
                    styles.primaryButton,
                    (outlookBusy || !outlookClientId.trim()) &&
                      styles.buttonDisabled,
                  ]}
                  onPress={configureOutlook}
                >
                  <Text style={styles.primaryButtonText}>
                    {outlookBusy
                      ? 'Enregistrement…'
                      : 'Enregistrer l’ID client'}
                  </Text>
                </Pressable>
              </View>
            ) : outlookStatus.connected ? (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Déconnecter Outlook"
                disabled={outlookBusy}
                style={[
                  styles.secondaryButton,
                  outlookBusy && styles.buttonDisabled,
                ]}
                onPress={disconnectOutlook}
              >
                <Text style={styles.secondaryButtonText}>
                  {outlookBusy ? 'Déconnexion…' : 'Déconnecter Outlook'}
                </Text>
              </Pressable>
            ) : (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Connecter Outlook"
                disabled={outlookBusy || !outlookStatus.configured}
                style={[
                  styles.primaryButton,
                  (outlookBusy || !outlookStatus.configured) &&
                    styles.buttonDisabled,
                ]}
                onPress={connectOutlook}
              >
                <Text style={styles.primaryButtonText}>
                  {outlookBusy ? 'Connexion…' : 'Connecter Outlook'}
                </Text>
              </Pressable>
            )}
          </View>
        ) : (
          <View style={styles.loadingRow}>
            <ActivityIndicator color="#38bdf8" />
            <Text style={styles.description}>Lecture du statut Outlook…</Text>
          </View>
        )}
        {outlookError ? (
          <Text style={styles.warningText}>{outlookError}</Text>
        ) : null}
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
  outlookDetail: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 18,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  warningText: {
    color: '#fca5a5',
    lineHeight: 19,
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
