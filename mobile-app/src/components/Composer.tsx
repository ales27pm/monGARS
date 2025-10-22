import React, { useState } from 'react';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { useVoiceAssistant } from '../hooks/useVoiceAssistant';

type Props = {
  onSend: (text: string, mode: 'chat' | 'embedding') => Promise<void>;
  sending: boolean;
};

const Composer: React.FC<Props> = ({ onSend, sending }) => {
  const [text, setText] = useState('');
  const [mode, setMode] = useState<'chat' | 'embedding'>('chat');
  const { listening, transcript, start, stop } = useVoiceAssistant(
    async (finalText) => {
      setText((prev) => (prev.length > 0 ? `${prev} ${finalText}` : finalText));
    },
  );

  const handleSubmit = async () => {
    if (!text.trim()) {
      return;
    }
    await onSend(text.trim(), mode);
    setText('');
  };

  return (
    <View style={styles.container}>
      <View style={styles.modeSwitch}>
        <Pressable
          accessibilityRole="button"
          onPress={() => setMode('chat')}
          style={[
            styles.modeButton,
            mode === 'chat' && styles.modeButtonActive,
          ]}
        >
          <Text style={styles.modeText}>Chat</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          onPress={() => setMode('embedding')}
          style={[
            styles.modeButton,
            mode === 'embedding' && styles.modeButtonActive,
          ]}
        >
          <Text style={styles.modeText}>Embedding</Text>
        </Pressable>
      </View>
      <TextInput
        accessibilityLabel="Prompt"
        multiline
        style={styles.input}
        value={transcript.length > 0 ? `${text}\n${transcript}` : text}
        onChangeText={setText}
        placeholder="Envoyez une instruction à MonGARS"
      />
      <View style={styles.actions}>
        <Pressable
          accessibilityRole="button"
          onPress={listening ? stop : start}
          style={[styles.actionButton, listening && styles.actionButtonActive]}
        >
          <Text style={styles.actionText}>{listening ? 'Stop' : 'Dicter'}</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          disabled={sending}
          onPress={handleSubmit}
          style={[styles.actionButton, sending && styles.actionButtonDisabled]}
        >
          <Text style={styles.actionText}>
            {sending ? 'Envoi…' : 'Envoyer'}
          </Text>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#020617',
  },
  input: {
    backgroundColor: '#0f172a',
    color: '#f8fafc',
    borderRadius: 12,
    padding: 12,
    minHeight: 120,
    textAlignVertical: 'top',
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  actionButton: {
    backgroundColor: '#1e40af',
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 12,
  },
  actionButtonDisabled: {
    opacity: 0.6,
  },
  actionButtonActive: {
    backgroundColor: '#0284c7',
  },
  actionText: {
    color: '#f8fafc',
    fontWeight: '600',
  },
  modeSwitch: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 12,
    gap: 12,
  },
  modeButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: '#1f2937',
  },
  modeButtonActive: {
    backgroundColor: '#2563eb',
  },
  modeText: {
    color: '#f8fafc',
    fontWeight: '500',
  },
});

export default Composer;
