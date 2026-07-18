import React from 'react';
import { render } from '@testing-library/react-native';
import MessageBubble from '../src/components/MessageBubble';

describe('MessageBubble', () => {
  it('shows a Core ML finish reason even when token metrics are absent', () => {
    const { getByText } = render(
      <MessageBubble
        message={{
          id: 'local-error',
          role: 'assistant',
          content: 'Réponse partielle',
          createdAt: new Date('2026-07-17T12:00:00Z'),
          metadata: {
            inferenceBackend: 'on-device',
            source: 'on-device',
            finishReason: 'error',
          },
        }}
      />,
    );

    expect(getByText('Core ML · error')).toBeTruthy();
  });
});
