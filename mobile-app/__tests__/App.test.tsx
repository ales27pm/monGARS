import React from 'react';
import { render } from '@testing-library/react-native';
import App from '../src/App';

describe('App', () => {
  it('renders the settings call-to-action when no session is present', async () => {
    const { findByText } = render(<App />);
    expect(
      await findByText(
        'Ouvrez les parametres pour recuperer un jeton et demarrer la conversation native.',
      ),
    ).toBeTruthy();
  });
});
