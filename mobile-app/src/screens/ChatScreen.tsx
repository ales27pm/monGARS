import React, { useCallback, useEffect } from 'react';
import {
  ActivityIndicator,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import type { ListRenderItem } from '@shopify/flash-list';
import { FlashList } from '@shopify/flash-list';
import Icon from 'react-native-vector-icons/Ionicons';
import MessageBubble from '../components/MessageBubble';
import Composer from '../components/Composer';
import { useChatStore } from '../store/chatStore';
import { useNavigation } from '@react-navigation/native';
import type { Message } from '../types';

type EmptyStateProps = {
  message: string;
  buttonLabel?: string;
  onButtonPress?: () => void;
};

const EmptyState: React.FC<EmptyStateProps> = ({
  message,
  buttonLabel,
  onButtonPress,
}) => {
  return (
    <View style={styles.emptyState}>
      <Text style={styles.emptyText}>{message}</Text>
      {buttonLabel ? (
        <Pressable style={styles.loginButton} onPress={onButtonPress}>
          <Text style={styles.loginButtonText}>{buttonLabel}</Text>
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
        <Icon name="wifi" size={22} color="#1e293b" />
      </Pressable>
      <Pressable accessibilityRole="button" onPress={onSettings}>
        <Icon name="settings-outline" size={22} color="#1e293b" />
      </Pressable>
    </View>
  );
};

const NoConversationPlaceholder: React.FC = () => (
  <EmptyState message="Commencez une nouvelle discussion avec MonGARS." />
);

const renderMessage: ListRenderItem<Message> = ({ item }) => (
  <MessageBubble message={item} />
);

const ChatScreen: React.FC = () => {
  const navigation = useNavigation();
  const {
    messages,
    initialize,
    loading,
    error,
    sendMessage,
    clearError,
    token,
  } = useChatStore();

  const headerRight = useCallback(
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
      headerRight,
    });
  }, [navigation, headerRight]);

  useEffect(() => {
    initialize().catch((err) => console.warn('init failed', err));
  }, [initialize, token]);

  const handleSend = useCallback(
    async (content: string, mode: 'chat' | 'embedding') => {
      await sendMessage(content, mode);
    },
    [sendMessage],
  );

  return (
    <SafeAreaView style={styles.container}>
      {error ? (
        <Pressable onPress={clearError} style={styles.banner}>
          <Text style={styles.bannerText}>{error}</Text>
        </Pressable>
      ) : null}
      {loading && token && (
        <View style={styles.loading}>
          <ActivityIndicator color="#2563eb" />
        </View>
      )}
      {!token ? (
        <EmptyState
          message="Connectez-vous dans les paramètres pour commencer à discuter."
          buttonLabel="Ouvrir les paramètres"
          onButtonPress={() => navigation.navigate('Settings' as never)}
        />
      ) : (
        <>
          <FlashList
            data={messages}
            contentContainerStyle={styles.list}
            estimatedItemSize={120}
            renderItem={renderMessage}
            ListEmptyComponent={NoConversationPlaceholder}
          />
          <Composer sending={loading} onSend={handleSend} />
        </>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#020617',
  },
  list: {
    paddingHorizontal: 16,
    paddingVertical: 24,
  },
  loading: {
    position: 'absolute',
    top: 16,
    right: 16,
  },
  banner: {
    backgroundColor: '#f97316',
    padding: 12,
    margin: 12,
    borderRadius: 10,
  },
  bannerText: {
    color: '#0f172a',
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    padding: 24,
    gap: 16,
  },
  emptyText: {
    color: '#94a3b8',
    textAlign: 'center',
  },
  headerActions: {
    flexDirection: 'row',
    gap: 16,
  },
  loginButton: {
    backgroundColor: '#1d4ed8',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  loginButtonText: {
    color: '#f8fafc',
    fontWeight: '600',
  },
});

export default ChatScreen;
