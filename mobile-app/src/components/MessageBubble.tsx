import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { Message } from '../types';
import { format } from 'date-fns';

interface Props {
  message: Message;
}

const MessageBubble: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';
  return (
    <View
      accessibilityLabel={`${message.role} message`}
      style={[
        styles.container,
        isUser ? styles.userContainer : styles.aiContainer,
      ]}
    >
      <Text style={[styles.text, isUser ? styles.userText : styles.aiText]}>
        {message.content}
      </Text>
      <Text style={styles.timestamp}>{format(message.createdAt, 'p')}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 18,
    marginVertical: 8,
    padding: 12,
    maxWidth: '85%',
  },
  userContainer: {
    alignSelf: 'flex-end',
    backgroundColor: '#2563eb',
  },
  aiContainer: {
    alignSelf: 'flex-start',
    backgroundColor: '#0f172a',
  },
  text: {
    fontSize: 16,
  },
  userText: {
    color: '#f8fafc',
  },
  aiText: {
    color: '#e2e8f0',
  },
  timestamp: {
    fontSize: 12,
    color: '#cbd5f5',
    marginTop: 6,
  },
});

export default MessageBubble;
