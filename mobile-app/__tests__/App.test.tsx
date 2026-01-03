import React from 'react';
import { render } from '@testing-library/react-native';
import App from '../src/App';

describe('App', () => {
  it('renders the initial login prompt when no token is present', async () => {
    const { findByText } = render(<App />);
    expect(
      await findByText(
        'Connectez-vous dans les paramètres pour commencer à discuter.',
      ),
    ).toBeTruthy();
  });
});
