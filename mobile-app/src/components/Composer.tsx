import React, { useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useVoiceAssistant } from '../hooks/useVoiceAssistant';
import { settings } from '../services/config';
import type { ChatMode, QuickAction } from '../types';

type Props = {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  onSend: (text: string) => Promise<void>;
  onDraftChange: (text: string) => void;
  quickActions: QuickAction[];
  sending: boolean;
};

const QUICK_ACTION_PRESETS: Record<QuickAction, string> = {
  code: 'Je souhaite écrire du code…',
  summarize: 'Résume la dernière conversation.',
  explain: 'Explique ta dernière réponse plus simplement.',
};

const QUICK_ACTION_LABELS: Record<QuickAction, string> = {
  code: 'Code',
  summarize: 'Résumer',
  explain: 'Expliquer',
};

const Composer: React.FC<Props> = ({
  mode,
  onModeChange,
  onSend,
  onDraftChange,
  quickActions,
  sending,
}) => {
  const [text, setText] = useState('');
  const [voiceAutoSend, setVoiceAutoSend] = useState(false);
  const { listening, transcript, start, stop } = useVoiceAssistant(
    async (finalText) => {
      const nextText = [text, finalText].filter(Boolean).join(' ').trim();
      setText(nextText);
      onDraftChange(nextText);
      if (voiceAutoSend && nextText) {
        try {
          await onSend(nextText);
          setText('');
          onDraftChange('');
        } catch (error) {
          console.warn('[Composer] auto-send failed', error);
        }
      }
    },
  );

  const handleTextChange = (value: string) => {
    setText(value);
    onDraftChange(value);
  };

  const handleSubmit = async (value = text) => {
    const nextValue = value.trim();
    if (!nextValue) {
      return;
    }

    await onSend(nextValue);
    setText('');
    onDraftChange('');
  };

  const handleQuickAction = async (action: QuickAction) => {
    const preset = QUICK_ACTION_PRESETS[action];
    setText(preset);
    onDraftChange(preset);
    await handleSubmit(preset);
  };

  return (
    <View style={styles.container}>
      <View style={styles.modeSwitch}>
        <Pressable
          accessibilityRole="button"
          onPress={() => onModeChange('chat')}
          style={[
            styles.modeButton,
            mode === 'chat' && styles.modeButtonActive,
          ]}
        >
          <Text
            style={[styles.modeText, mode === 'chat' && styles.modeTextActive]}
          >
            Chat
          </Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          onPress={() => onModeChange('embed')}
          style={[
            styles.modeButton,
            mode === 'embed' && styles.modeButtonActive,
          ]}
        >
          <Text
            style={[styles.modeText, mode === 'embed' && styles.modeTextActive]}
          >
            Embedding
          </Text>
        </Pressable>
      </View>

      <TextInput
        accessibilityLabel="Prompt"
        multiline
        style={styles.input}
        value={transcript.length > 0 ? `${text}\n${transcript}` : text}
        onChangeText={handleTextChange}
        placeholder={
          mode === 'embed'
            ? 'Entrez le texte a encoder'
            : 'Envoyez une instruction a monGARS'
        }
        placeholderTextColor="#64748b"
      />

      <View style={styles.metaRow}>
        <Text style={styles.helper}>
          {mode === 'embed'
            ? 'Renvoie un resume de vecteurs.'
            : 'Reponse LLM avec synchro temps reel.'}
        </Text>
        <Text style={styles.counter}>{text.trim().length}/1000</Text>
      </View>

      <View style={styles.quickActions}>
        {quickActions.map((action) => (
          <Pressable
            key={action}
            onPress={() => handleQuickAction(action)}
            style={styles.quickAction}
          >
            <Text style={styles.quickActionText}>
              {QUICK_ACTION_LABELS[action]}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.voiceCard}>
        <View style={styles.voiceHeader}>
          <Text style={styles.voiceTitle}>Dictee vocale</Text>
          <Text style={styles.voiceCaption}>{settings.voiceLocale}</Text>
        </View>
        <View style={styles.voiceControls}>
          <Pressable
            accessibilityRole="button"
            onPress={listening ? stop : start}
            style={[styles.voiceButton, listening && styles.voiceButtonActive]}
          >
            <Text style={styles.voiceButtonText}>
              {listening ? 'Arreter' : 'Dicter'}
            </Text>
          </Pressable>
          <View style={styles.switchRow}>
            <Text style={styles.switchLabel}>Envoi auto</Text>
            <Switch
              value={voiceAutoSend}
              onValueChange={setVoiceAutoSend}
              trackColor={{ false: '#334155', true: '#f59e0b' }}
              thumbColor="#f8fafc"
            />
          </View>
        </View>
        {transcript ? (
          <Text style={styles.transcriptPreview}>{transcript}</Text>
        ) : (
          <Text style={styles.voiceHint}>
            Le texte intermediaire apparait ici pendant la dictee.
          </Text>
        )}
      </View>

      <View style={styles.actions}>
        <Pressable
          accessibilityRole="button"
          disabled={sending}
          onPress={() => handleSubmit()}
          style={[styles.sendButton, sending && styles.sendButtonDisabled]}
        >
          <Text style={styles.sendButtonText}>
            {sending ? 'Envoi…' : mode === 'embed' ? 'Generer' : 'Envoyer'}
          </Text>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 14,
  },
  modeSwitch: {
    flexDirection: 'row',
    gap: 10,
  },
  modeButton: {
    flex: 1,
    borderRadius: 999,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#172033',
    borderWidth: 1,
    borderColor: '#22304a',
    alignItems: 'center',
  },
  modeButtonActive: {
    backgroundColor: '#f59e0b',
    borderColor: '#fbbf24',
  },
  modeText: {
    color: '#cbd5e1',
    fontWeight: '600',
  },
  modeTextActive: {
    color: '#111827',
  },
  input: {
    minHeight: 132,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 16,
    backgroundColor: '#08111f',
    color: '#f8fafc',
    textAlignVertical: 'top',
    borderWidth: 1,
    borderColor: '#1f2d45',
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  helper: {
    flex: 1,
    color: '#94a3b8',
    fontSize: 12,
  },
  counter: {
    color: '#fbbf24',
    fontSize: 12,
    fontWeight: '600',
  },
  quickActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  quickAction: {
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#111b2d',
    borderWidth: 1,
    borderColor: '#23314d',
  },
  quickActionText: {
    color: '#dbeafe',
    fontWeight: '600',
  },
  voiceCard: {
    borderRadius: 20,
    padding: 16,
    backgroundColor: '#10192b',
    borderWidth: 1,
    borderColor: '#22304a',
    gap: 12,
  },
  voiceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  voiceTitle: {
    color: '#f8fafc',
    fontSize: 15,
    fontWeight: '700',
  },
  voiceCaption: {
    color: '#94a3b8',
    fontSize: 12,
  },
  voiceControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  voiceButton: {
    borderRadius: 14,
    paddingHorizontal: 18,
    paddingVertical: 12,
    backgroundColor: '#1d4ed8',
  },
  voiceButtonActive: {
    backgroundColor: '#b91c1c',
  },
  voiceButtonText: {
    color: '#f8fafc',
    fontWeight: '700',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  switchLabel: {
    color: '#cbd5e1',
    fontSize: 13,
  },
  transcriptPreview: {
    color: '#fbbf24',
    fontStyle: 'italic',
  },
  voiceHint: {
    color: '#64748b',
    fontSize: 12,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  sendButton: {
    minWidth: 144,
    borderRadius: 18,
    paddingHorizontal: 20,
    paddingVertical: 14,
    backgroundColor: '#f97316',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.6,
  },
  sendButtonText: {
    color: '#fff7ed',
    fontWeight: '800',
    letterSpacing: 0.3,
  },
});

export default Composer;
