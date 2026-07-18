import React from 'react';
import { fireEvent, render } from '@testing-library/react-native';
import App from '../src/App';
import { useChatStore } from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';

jest.mock('@shopify/flash-list', () => ({
  FlashList: require('react-native').FlatList,
}));

describe('App', () => {
  it('renders the settings call-to-action when no session is present', async () => {
    const { findByText } = render(<App />);
    expect(
      await findByText(
        'Ouvrez les parametres pour recuperer un jeton et demarrer la conversation native.',
      ),
    ).toBeTruthy();
  });

  it('does not replace persisted local history with the unavailable-model empty state', () => {
    useInferenceStore.setState({
      backend: 'on-device',
      status: {
        phase: 'unavailable',
        modelId: null,
        displayName: null,
        revision: null,
        installedBytes: 0,
        contextLength: 0,
        minimumIOSVersion: 18,
        detail: 'Module Core ML indisponible',
      },
    });
    useChatStore.setState({
      session: null,
      initialize: jest.fn().mockResolvedValue(undefined),
      messages: [
        {
          id: 'persisted-local-message',
          role: 'assistant',
          content: 'Conversation locale conservée',
          createdAt: new Date(),
          metadata: {
            inferenceBackend: 'on-device',
            source: 'on-device',
            localOwnerId: 'guest',
            finishReason: 'eos',
          },
        },
      ],
    });

    const { getByPlaceholderText, getByText, queryByText } = render(<App />);

    expect(getByText('Conversation')).toBeTruthy();
    expect(getByText('Conversation locale conservée')).toBeTruthy();
    expect(queryByText('Modele local requis')).toBeNull();

    fireEvent.changeText(
      getByPlaceholderText('Filtrer les messages'),
      'aucune-correspondance',
    );

    expect(getByText('Aucun message ne correspond au filtre.')).toBeTruthy();
    expect(queryByText('Modele local requis')).toBeNull();
  });
});
