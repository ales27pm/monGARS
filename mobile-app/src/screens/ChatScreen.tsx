import React, {
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from 'react';
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import type { ListRenderItem } from '@shopify/flash-list';
import { FlashList } from '@shopify/flash-list';
import Icon from 'react-native-vector-icons/Ionicons';
import { useNavigation } from '@react-navigation/native';
import Composer from '../components/Composer';
import MessageBubble from '../components/MessageBubble';
import { settings } from '../services/config';
import {
  getLocalConversationOwner,
  isConversationMessageVisible,
  useChatStore,
} from '../store/chatStore';
import { useInferenceStore } from '../store/inferenceStore';
import type { Message } from '../types';
import { summarizeAgentApprovalArguments } from '../agent/approvalSummary';
import {
  appIntentHandoffPreview,
  appIntentHandoffTitle,
} from '../agent/appIntentHandoff';
import { subscribeToAgentTriggerResume } from '../services/agentTriggerResume';
import { subscribeToAppIntentResume } from '../services/appIntentResume';
import { buildJsonExport, buildMarkdownExport } from '../utils/conversation';

const EmptyState: React.FC<{
  title: string;
  description: string;
  buttonLabel?: string;
  onButtonPress?: () => void;
}> = ({ title, description, buttonLabel, onButtonPress }) => {
  return (
    <View style={styles.emptyState}>
      <Text style={styles.emptyTitle}>{title}</Text>
      <Text style={styles.emptyText}>{description}</Text>
      {buttonLabel ? (
        <Pressable style={styles.emptyButton} onPress={onButtonPress}>
          <Text style={styles.emptyButtonText}>{buttonLabel}</Text>
        </Pressable>
      ) : null}
    </View>
  );
};

type HeaderActionsProps = {
  onDiagnostics: () => void;
  onSettings: () => void;
};

const HeaderActions: React.FC<HeaderActionsProps> = ({
  onDiagnostics,
  onSettings,
}) => {
  return (
    <View style={styles.headerActions}>
      <Pressable accessibilityRole="button" onPress={onDiagnostics}>
        <Icon name="analytics-outline" size={22} color="#f8fafc" />
      </Pressable>
      <Pressable accessibilityRole="button" onPress={onSettings}>
        <Icon name="settings-outline" size={22} color="#f8fafc" />
      </Pressable>
    </View>
  );
};

const ChatScreen: React.FC = () => {
  const navigation = useNavigation();
  const [draft, setDraft] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const deferredSearchQuery = useDeferredValue(searchQuery);
  const {
    session,
    messages,
    loading,
    historyLoading,
    error,
    notice,
    mode,
    quickActions,
    connection,
    pendingAgentApproval,
    pendingAgentPermission,
    pendingAgentTrigger,
    pendingAppIntentHandoff,
    initialize,
    refreshPendingAgentTrigger,
    refreshPendingAppIntentHandoff,
    sendMessage,
    approvePendingAgent,
    rejectPendingAgent,
    requestPendingAgentPermission,
    dismissPendingAgentPermission,
    runPendingAgentTrigger,
    dismissPendingAgentTrigger,
    runPendingAppIntentHandoff,
    dismissPendingAppIntentHandoff,
    requestQuickActions,
    setMode,
    clearError,
    clearNotice,
    retryRealtime,
    cancelGeneration,
  } = useChatStore();
  const {
    backend,
    status: localStatus,
    progress: localProgress,
    generation: localGeneration,
    lastResult: localResult,
  } = useInferenceStore();

  const renderHeaderActions = useCallback(
    () => (
      <HeaderActions
        onDiagnostics={() => navigation.navigate('Diagnostics' as never)}
        onSettings={() => navigation.navigate('Settings' as never)}
      />
    ),
    [navigation],
  );

  useEffect(() => {
    navigation.setOptions({
      headerStyle: {
        backgroundColor: '#0f172a',
      },
      headerTintColor: '#f8fafc',
      headerTitleStyle: {
        fontWeight: '700',
      },
      headerRight: renderHeaderActions,
    });
  }, [navigation, renderHeaderActions]);

  useEffect(() => {
    initialize().catch((err) => console.warn('[ChatScreen] init failed', err));
  }, [initialize, session?.token, session?.username]);

  useEffect(
    () => subscribeToAgentTriggerResume(refreshPendingAgentTrigger),
    [refreshPendingAgentTrigger],
  );

  useEffect(
    () => subscribeToAppIntentResume(refreshPendingAppIntentHandoff),
    [refreshPendingAppIntentHandoff],
  );

  useEffect(() => {
    if (backend === 'on-device') {
      return;
    }
    const timer = setTimeout(() => {
      requestQuickActions(draft).catch((suggestionError) => {
        console.warn('[ChatScreen] suggestions failed', suggestionError);
      });
    }, 220);

    return () => clearTimeout(timer);
  }, [backend, draft, requestQuickActions, mode]);

  const activeMessages = useMemo(() => {
    const localOwnerId = getLocalConversationOwner(session);
    return messages.filter((message) =>
      isConversationMessageVisible(message, backend, localOwnerId),
    );
  }, [backend, messages, session]);

  const filteredMessages = useMemo(() => {
    const query = deferredSearchQuery.trim().toLowerCase();
    if (!query) {
      return activeMessages;
    }

    return activeMessages.filter((message) =>
      message.content.toLowerCase().includes(query),
    );
  }, [activeMessages, deferredSearchQuery]);

  const renderMessage: ListRenderItem<Message> = useCallback(
    ({ item }) => (
      <MessageBubble
        message={item}
        highlight={deferredSearchQuery.trim() || undefined}
      />
    ),
    [deferredSearchQuery],
  );

  const handleShare = async (format: 'markdown' | 'json') => {
    if (!activeMessages.length) {
      return;
    }

    await Share.share({
      title: 'monGARS export',
      message:
        format === 'json'
          ? buildJsonExport(activeMessages)
          : buildMarkdownExport(activeMessages),
    });
  };

  const handleSend = async (text: string) => {
    await sendMessage(text, mode);
  };

  const showReconnect =
    backend === 'server' &&
    session &&
    (connection.status === 'error' ||
      connection.status === 'offline' ||
      connection.status === 'auth-required');
  const localReady =
    localStatus.phase === 'ready' || localStatus.phase === 'generating';
  const activeStatus =
    backend === 'on-device' ? localStatus.phase : connection.status;
  const localTokensPerSecond =
    localGeneration?.tokensPerSecond ?? localResult?.tokensPerSecond ?? 0;
  const pendingApprovalSummary = useMemo(
    () =>
      pendingAgentApproval
        ? summarizeAgentApprovalArguments(pendingAgentApproval.arguments)
        : [],
    [pendingAgentApproval],
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.backdrop}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.heroCard}>
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.eyebrow}>
                {backend === 'on-device'
                  ? 'Core ML · sur l appareil'
                  : 'Operator Console · serveur'}
              </Text>
              <Text style={styles.heroTitle}>monGARS mobile</Text>
              <Text style={styles.heroSubtitle}>
                {backend === 'on-device'
                  ? 'Modèle et chat locaux; les outils choisis peuvent joindre leur fournisseur.'
                  : 'Chat natif, synchro temps reel, recherche et export.'}
              </Text>
            </View>
            <View
              style={[
                styles.statusPill,
                ['online', 'ready', 'generating'].includes(activeStatus)
                  ? styles.statusOnline
                  : [
                        'connecting',
                        'downloading',
                        'verifying',
                        'compiling',
                        'loading',
                      ].includes(activeStatus)
                    ? styles.statusConnecting
                    : styles.statusOffline,
              ]}
            >
              <Text style={styles.statusPillText}>{activeStatus}</Text>
            </View>
          </View>
          <View style={styles.metricsRow}>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>
                {backend === 'on-device' ? 'Modele' : 'Session'}
              </Text>
              <Text style={styles.metricValue}>
                {backend === 'on-device'
                  ? (localStatus.displayName ?? 'Core ML')
                  : (session?.username ?? 'Aucune')}
              </Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>
                {backend === 'on-device' ? 'Contexte' : 'Dernier message'}
              </Text>
              <Text style={styles.metricValue}>
                {backend === 'on-device'
                  ? localStatus.contextLength
                    ? `${localStatus.contextLength} jetons`
                    : '—'
                  : connection.lastMessageAt
                    ? connection.lastMessageAt.toLocaleTimeString()
                    : '—'}
              </Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricLabel}>
                {backend === 'on-device' ? 'Debit' : 'Latence'}
              </Text>
              <Text style={styles.metricValue}>
                {backend === 'on-device'
                  ? localTokensPerSecond > 0
                    ? `${localTokensPerSecond.toFixed(1)} j/s`
                    : '—'
                  : connection.latencyMs
                    ? `${connection.latencyMs} ms`
                    : '—'}
              </Text>
            </View>
          </View>
          <Text style={styles.connectionDetail}>
            {backend === 'on-device'
              ? localProgress
                ? `${localProgress.detail ?? 'Preparation du modele'} · ${Math.round(
                    localProgress.fractionCompleted * 100,
                  )}%`
                : (localStatus.detail ??
                  'Backend local selectionne; aucun appel au serveur de chat.')
              : (connection.detail ?? 'En attente de connexion.')}
          </Text>
          {showReconnect ? (
            <Pressable style={styles.reconnectButton} onPress={retryRealtime}>
              <Text style={styles.reconnectButtonText}>Reconnecter</Text>
            </Pressable>
          ) : null}
        </View>

        {error ? (
          <Pressable onPress={clearError} style={styles.bannerError}>
            <Text style={styles.bannerTitle}>Erreur</Text>
            <Text style={styles.bannerText}>{error}</Text>
          </Pressable>
        ) : null}

        {notice ? (
          <Pressable onPress={clearNotice} style={styles.bannerNotice}>
            <Text style={styles.bannerTitle}>Etat</Text>
            <Text style={styles.bannerText}>{notice.message}</Text>
          </Pressable>
        ) : null}

        {pendingAppIntentHandoff ? (
          <View style={styles.shortcutCard}>
            <Text style={styles.shortcutEyebrow}>
              Siri · Raccourcis · {pendingAppIntentHandoff.profileLabel}
            </Text>
            <Text selectable style={styles.shortcutDetail}>
              Profil actif : {pendingAppIntentHandoff.ownerId}
            </Text>
            <Text style={styles.approvalTitle}>
              {appIntentHandoffTitle(pendingAppIntentHandoff)}
            </Text>
            <Text selectable style={styles.shortcutInput}>
              {appIntentHandoffPreview(
                pendingAppIntentHandoff,
                pendingAppIntentHandoff.resolvedTrigger,
              )}
            </Text>
            <Text style={styles.shortcutDetail}>
              {!pendingAppIntentHandoff.profileMatches
                ? 'Le contenu et le type restent masqués pour le profil actif; vous pouvez seulement ignorer cette demande exacte.'
                : pendingAppIntentHandoff.kind === 'runTrigger' &&
                    !pendingAppIntentHandoff.resolvedTrigger
                  ? 'Aucune requête exacte ne peut être prévisualisée; l’exécution reste bloquée.'
                  : pendingAppIntentHandoff.kind === 'diagnostics'
                    ? 'Cette action ouvre seulement l écran passif dans l app.'
                    : backend === 'on-device'
                      ? 'Rien ne sera exécuté avant votre confirmation au premier plan.'
                      : 'La mémoire reste locale; les requêtes de modèle exigent le backend local.'}
            </Text>
            <View style={styles.approvalActions}>
              {pendingAppIntentHandoff.profileMatches &&
              (pendingAppIntentHandoff.kind !== 'runTrigger' ||
                pendingAppIntentHandoff.resolvedTrigger) ? (
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Confirmer l action Siri ou Raccourcis"
                  disabled={loading}
                  style={[
                    styles.approvalButton,
                    styles.approveButton,
                    loading ? styles.disabledButton : null,
                  ]}
                  onPress={() => {
                    runPendingAppIntentHandoff()
                      .then((destination) => {
                        if (destination === 'diagnostics') {
                          navigation.navigate('Diagnostics' as never);
                        }
                      })
                      .catch((handoffError) =>
                        console.warn(
                          '[ChatScreen] App Intent handoff failed',
                          handoffError,
                        ),
                      );
                  }}
                >
                  <Text style={styles.approveButtonText}>
                    {pendingAppIntentHandoff.kind === 'diagnostics'
                      ? 'Ouvrir'
                      : 'Exécuter'}
                  </Text>
                </Pressable>
              ) : null}
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Ignorer l action Siri ou Raccourcis"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.rejectButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={dismissPendingAppIntentHandoff}
              >
                <Text style={styles.rejectButtonText}>Ignorer</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        {backend === 'on-device' && pendingAgentTrigger ? (
          <View style={styles.triggerCard}>
            <Text style={styles.triggerEyebrow}>Requête planifiée</Text>
            <Text style={styles.approvalTitle}>
              {pendingAgentTrigger.title}
            </Text>
            <Text style={styles.approvalText}>
              Prompt exact qui sera transmis à l agent :
            </Text>
            <Text selectable style={styles.triggerPrompt}>
              {pendingAgentTrigger.prompt}
            </Text>
            <Text style={styles.approvalText}>
              {pendingAgentTrigger.repeats
                ? 'Déclencheur récurrent. Le lancement reste manuel.'
                : 'Déclencheur ponctuel. Aucun agent ne part sans votre action.'}
            </Text>
            <View style={styles.approvalActions}>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Lancer la requête planifiée"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.approveButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={() => {
                  runPendingAgentTrigger().catch((triggerError) =>
                    console.warn(
                      '[ChatScreen] scheduled agent failed',
                      triggerError,
                    ),
                  );
                }}
              >
                <Text style={styles.approveButtonText}>Lancer</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Ignorer la requête planifiée"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.rejectButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={dismissPendingAgentTrigger}
              >
                <Text style={styles.rejectButtonText}>Ignorer</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        {backend === 'on-device' && pendingAgentApproval ? (
          <View style={styles.approvalCard}>
            <Text style={styles.approvalEyebrow}>
              Approbation locale requise
            </Text>
            <Text style={styles.approvalTitle}>
              {pendingAgentApproval.displayName}
            </Text>
            <Text style={styles.approvalText}>
              L outil « {pendingAgentApproval.toolId} » ne sera pas exécuté
              avant votre approbation explicite. Risque :{' '}
              {pendingAgentApproval.risk}.
            </Text>
            <View style={styles.approvalArguments}>
              {pendingApprovalSummary.length ? (
                pendingApprovalSummary.map((argument) => (
                  <View key={argument.key} style={styles.approvalArgumentRow}>
                    <Text style={styles.approvalArgumentKey}>
                      {argument.key}
                    </Text>
                    <Text selectable style={styles.approvalArgumentValue}>
                      {argument.value}
                    </Text>
                  </View>
                ))
              ) : (
                <Text style={styles.approvalArgumentValue}>Aucun argument</Text>
              )}
            </View>
            <View style={styles.approvalActions}>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Approuver l action locale"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.approveButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={() => {
                  approvePendingAgent().catch((approvalError) =>
                    console.warn(
                      '[ChatScreen] agent approval failed',
                      approvalError,
                    ),
                  );
                }}
              >
                <Text style={styles.approveButtonText}>Approuver</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Rejeter l action locale"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.rejectButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={() => {
                  rejectPendingAgent().catch((rejectionError) =>
                    console.warn(
                      '[ChatScreen] agent rejection failed',
                      rejectionError,
                    ),
                  );
                }}
              >
                <Text style={styles.rejectButtonText}>Rejeter</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        {backend === 'on-device' && pendingAgentPermission ? (
          <View style={styles.approvalCard}>
            <Text style={styles.approvalEyebrow}>Autorisation iOS requise</Text>
            <Text style={styles.approvalTitle}>
              {pendingAgentPermission.permission}
            </Text>
            <Text style={styles.approvalText}>
              iOS doit afficher sa demande d accès avant que l agent puisse
              utiliser cette capacité. Refuser ou fermer la demande n exécute
              aucun outil.
            </Text>
            <View style={styles.approvalActions}>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Demander l autorisation iOS"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.approveButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={() => {
                  requestPendingAgentPermission().catch((permissionError) =>
                    console.warn(
                      '[ChatScreen] agent permission failed',
                      permissionError,
                    ),
                  );
                }}
              >
                <Text style={styles.approveButtonText}>Continuer dans iOS</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Annuler la demande d autorisation"
                disabled={loading}
                style={[
                  styles.approvalButton,
                  styles.rejectButton,
                  loading ? styles.disabledButton : null,
                ]}
                onPress={dismissPendingAgentPermission}
              >
                <Text style={styles.rejectButtonText}>Annuler</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        <View style={styles.searchCard}>
          <Text style={styles.sectionTitle}>Recherche</Text>
          <View style={styles.searchRow}>
            <TextInput
              value={searchQuery}
              onChangeText={(value) => {
                startTransition(() => setSearchQuery(value));
              }}
              placeholder="Filtrer les messages"
              placeholderTextColor="#64748b"
              style={styles.searchInput}
            />
            <Pressable
              style={styles.clearSearchButton}
              onPress={() => setSearchQuery('')}
            >
              <Text style={styles.clearSearchText}>Effacer</Text>
            </Pressable>
          </View>
          <Text style={styles.searchHint}>
            {deferredSearchQuery.trim()
              ? `${filteredMessages.length} message(s) correspondent.`
              : 'Filtrez la conversation active.'}
          </Text>
        </View>

        <View style={styles.transcriptCard}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Conversation</Text>
            <View style={styles.exportRow}>
              <Pressable
                style={styles.exportButton}
                onPress={() => handleShare('markdown')}
              >
                <Text style={styles.exportButtonText}>Markdown</Text>
              </Pressable>
              <Pressable
                style={styles.exportButton}
                onPress={() => handleShare('json')}
              >
                <Text style={styles.exportButtonText}>JSON</Text>
              </Pressable>
            </View>
          </View>

          {backend === 'server' && historyLoading ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color="#f59e0b" />
              <Text style={styles.loadingText}>
                Chargement de l historique…
              </Text>
            </View>
          ) : null}

          {backend === 'on-device' &&
          !localReady &&
          activeMessages.length === 0 ? (
            <EmptyState
              title="Modele local requis"
              description={
                localStatus.detail ??
                'Telechargez et verifiez le modele Core ML dans les parametres.'
              }
              buttonLabel="Gerer le modele"
              onButtonPress={() => navigation.navigate('Settings' as never)}
            />
          ) : backend === 'server' && !session ? (
            <EmptyState
              title="Connexion requise"
              description="Ouvrez les parametres pour recuperer un jeton et demarrer la conversation native."
              buttonLabel="Ouvrir les parametres"
              onButtonPress={() => navigation.navigate('Settings' as never)}
            />
          ) : filteredMessages.length === 0 ? (
            <EmptyState
              title="Conversation vide"
              description={
                deferredSearchQuery.trim()
                  ? 'Aucun message ne correspond au filtre.'
                  : 'Commencez une nouvelle discussion avec monGARS.'
              }
            />
          ) : (
            <FlashList
              data={filteredMessages}
              contentContainerStyle={styles.list}
              estimatedItemSize={140}
              renderItem={renderMessage}
            />
          )}
        </View>

        <View style={styles.composerCard}>
          <Text style={styles.sectionTitle}>Composer</Text>
          <Composer
            backend={backend}
            mode={mode}
            onModeChange={(nextMode) => {
              if (
                nextMode === 'embed' &&
                (backend === 'on-device' || !settings.embedServiceUrl)
              ) {
                clearNotice();
                useChatStore.setState({
                  notice: {
                    tone: 'warning',
                    message:
                      backend === 'on-device'
                        ? 'Les embeddings exigent le backend serveur.'
                        : 'Service d embedding indisponible.',
                  },
                });
                return;
              }
              setMode(nextMode);
            }}
            onSend={handleSend}
            onDraftChange={setDraft}
            quickActions={quickActions}
            sending={loading}
            canSend={
              backend === 'on-device'
                ? localReady && !pendingAgentApproval && !pendingAgentPermission
                : Boolean(session)
            }
            embeddingAvailable={
              backend === 'server' && Boolean(settings.embedServiceUrl)
            }
            onCancel={backend === 'on-device' ? cancelGeneration : undefined}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#060b16',
  },
  backdrop: {
    flex: 1,
  },
  scrollContent: {
    padding: 18,
    gap: 18,
  },
  heroCard: {
    borderRadius: 30,
    padding: 22,
    backgroundColor: '#10192b',
    borderWidth: 1,
    borderColor: '#24324d',
    gap: 16,
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  eyebrow: {
    color: '#f59e0b',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
    fontSize: 12,
    fontWeight: '700',
  },
  heroTitle: {
    color: '#f8fafc',
    fontSize: 28,
    fontWeight: '800',
  },
  heroSubtitle: {
    color: '#94a3b8',
    marginTop: 6,
    lineHeight: 20,
  },
  statusPill: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  statusOnline: {
    backgroundColor: '#14532d',
  },
  statusConnecting: {
    backgroundColor: '#78350f',
  },
  statusOffline: {
    backgroundColor: '#7f1d1d',
  },
  statusPillText: {
    color: '#f8fafc',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  metricsRow: {
    flexDirection: 'row',
    gap: 10,
  },
  metricCard: {
    flex: 1,
    borderRadius: 18,
    padding: 14,
    backgroundColor: '#0a1220',
  },
  metricLabel: {
    color: '#64748b',
    fontSize: 12,
    marginBottom: 6,
  },
  metricValue: {
    color: '#f8fafc',
    fontWeight: '700',
  },
  connectionDetail: {
    color: '#cbd5e1',
  },
  reconnectButton: {
    alignSelf: 'flex-start',
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#f97316',
  },
  reconnectButtonText: {
    color: '#fff7ed',
    fontWeight: '700',
  },
  bannerError: {
    borderRadius: 18,
    padding: 16,
    backgroundColor: '#7f1d1d',
  },
  bannerNotice: {
    borderRadius: 18,
    padding: 16,
    backgroundColor: '#172554',
  },
  bannerTitle: {
    color: '#f8fafc',
    fontWeight: '800',
    marginBottom: 4,
  },
  bannerText: {
    color: '#e2e8f0',
  },
  approvalCard: {
    borderRadius: 22,
    padding: 18,
    backgroundColor: '#2a1b08',
    borderWidth: 1,
    borderColor: '#f59e0b',
    gap: 10,
  },
  triggerCard: {
    borderRadius: 22,
    padding: 18,
    backgroundColor: '#10233a',
    borderWidth: 1,
    borderColor: '#38bdf8',
    gap: 10,
  },
  shortcutCard: {
    borderRadius: 22,
    padding: 18,
    backgroundColor: '#172139',
    borderWidth: 1,
    borderColor: '#a78bfa',
    gap: 10,
  },
  shortcutEyebrow: {
    color: '#c4b5fd',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  shortcutInput: {
    color: '#ede9fe',
    lineHeight: 21,
  },
  shortcutDetail: {
    color: '#ddd6fe',
    lineHeight: 21,
  },
  triggerEyebrow: {
    color: '#7dd3fc',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  triggerPrompt: {
    color: '#e0f2fe',
    lineHeight: 21,
  },
  approvalEyebrow: {
    color: '#fbbf24',
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  approvalTitle: {
    color: '#fff7ed',
    fontSize: 18,
    fontWeight: '800',
  },
  approvalText: {
    color: '#fed7aa',
    lineHeight: 21,
  },
  approvalActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 4,
  },
  approvalArguments: {
    borderRadius: 14,
    padding: 12,
    backgroundColor: '#170f05',
    gap: 8,
  },
  approvalArgumentRow: {
    gap: 2,
  },
  approvalArgumentKey: {
    color: '#fbbf24',
    fontSize: 12,
    fontWeight: '700',
  },
  approvalArgumentValue: {
    color: '#fff7ed',
    lineHeight: 20,
  },
  approvalButton: {
    flex: 1,
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 12,
    alignItems: 'center',
  },
  approveButton: {
    backgroundColor: '#f59e0b',
  },
  rejectButton: {
    backgroundColor: '#3f1d1d',
    borderWidth: 1,
    borderColor: '#ef4444',
  },
  approveButtonText: {
    color: '#111827',
    fontWeight: '800',
  },
  rejectButtonText: {
    color: '#fecaca',
    fontWeight: '800',
  },
  disabledButton: {
    opacity: 0.55,
  },
  searchCard: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: '#0d1525',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 12,
  },
  searchRow: {
    flexDirection: 'row',
    gap: 10,
  },
  searchInput: {
    flex: 1,
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: '#07101d',
    color: '#f8fafc',
    borderWidth: 1,
    borderColor: '#22304a',
  },
  clearSearchButton: {
    borderRadius: 16,
    paddingHorizontal: 14,
    justifyContent: 'center',
    backgroundColor: '#172033',
  },
  clearSearchText: {
    color: '#dbeafe',
    fontWeight: '600',
  },
  searchHint: {
    color: '#94a3b8',
  },
  transcriptCard: {
    minHeight: 360,
    borderRadius: 28,
    padding: 18,
    backgroundColor: '#0b1322',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 14,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
    alignItems: 'center',
  },
  sectionTitle: {
    color: '#f8fafc',
    fontSize: 17,
    fontWeight: '800',
  },
  exportRow: {
    flexDirection: 'row',
    gap: 8,
  },
  exportButton: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#172033',
  },
  exportButtonText: {
    color: '#dbeafe',
    fontWeight: '700',
    fontSize: 12,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  loadingText: {
    color: '#cbd5e1',
  },
  list: {
    paddingBottom: 24,
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 30,
    gap: 12,
  },
  emptyTitle: {
    color: '#f8fafc',
    fontSize: 18,
    fontWeight: '700',
  },
  emptyText: {
    color: '#94a3b8',
    textAlign: 'center',
    lineHeight: 21,
  },
  emptyButton: {
    borderRadius: 16,
    paddingHorizontal: 18,
    paddingVertical: 12,
    backgroundColor: '#f59e0b',
  },
  emptyButtonText: {
    color: '#111827',
    fontWeight: '800',
  },
  composerCard: {
    borderRadius: 28,
    padding: 18,
    backgroundColor: '#0d1525',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 14,
  },
  headerActions: {
    flexDirection: 'row',
    gap: 16,
  },
});

export default ChatScreen;
