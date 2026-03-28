import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { format } from 'date-fns';
import type { Message } from '../types';

type Props = {
  message: Message;
  highlight?: string;
};

function renderHighlighted(content: string, highlight?: string) {
  const token = highlight?.trim();
  if (!token) {
    return <Text style={styles.body}>{content}</Text>;
  }

  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const parts = content.split(new RegExp(`(${escaped})`, 'gi'));

  return (
    <Text style={styles.body}>
      {parts.map((part, index) => {
        const match = part.toLowerCase() === token.toLowerCase();
        return (
          <Text
            key={`${part}-${index}`}
            style={match ? styles.match : undefined}
          >
            {part}
          </Text>
        );
      })}
    </Text>
  );
}

const MessageBubble: React.FC<Props> = ({ message, highlight }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const roleLabel = isUser ? 'Operateur' : isSystem ? 'Systeme' : 'monGARS';

  return (
    <View
      accessibilityLabel={`${message.role} message`}
      style={[
        styles.container,
        isUser
          ? styles.userContainer
          : isSystem
            ? styles.systemContainer
            : styles.aiContainer,
      ]}
    >
      <View style={styles.header}>
        <Text style={styles.role}>{roleLabel}</Text>
        <Text style={styles.timestamp}>{format(message.createdAt, 'p')}</Text>
      </View>
      {renderHighlighted(message.content, highlight)}
      {message.metadata?.confidence !== undefined ? (
        <Text style={styles.meta}>
          Confiance {message.metadata.confidence.toFixed(2)} · Temps{' '}
          {message.metadata.processingTime?.toFixed(2) ?? '0.00'}s
        </Text>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 24,
    marginVertical: 8,
    padding: 16,
    maxWidth: '88%',
    borderWidth: 1,
  },
  userContainer: {
    alignSelf: 'flex-end',
    backgroundColor: '#f59e0b',
    borderColor: '#fbbf24',
  },
  aiContainer: {
    alignSelf: 'flex-start',
    backgroundColor: '#0f172a',
    borderColor: '#22304a',
  },
  systemContainer: {
    alignSelf: 'center',
    backgroundColor: '#10261c',
    borderColor: '#1f6b45',
    maxWidth: '94%',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
    gap: 10,
  },
  role: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1,
    textTransform: 'uppercase',
    color: '#e2e8f0',
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: '#f8fafc',
  },
  match: {
    backgroundColor: '#7c2d12',
    color: '#ffedd5',
  },
  timestamp: {
    fontSize: 12,
    color: '#cbd5e1',
  },
  meta: {
    marginTop: 10,
    fontSize: 11,
    color: '#dbeafe',
  },
});

export default MessageBubble;
