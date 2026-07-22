import { shouldUseNativeAgent } from '../src/services/onDeviceAgentService';

describe('on-device agent routing', () => {
  it.each([
    'What is the weather in Toronto?',
    'Search web for Swift concurrency',
    'Set an alarm for 8 am',
    'Take a photo',
    'Draft email',
  ])(
    'routes tool or clarification prompt through native policy: %s',
    (prompt) => {
      expect(shouldUseNativeAgent(prompt)).toBe(true);
    },
  );

  it.each(['Bonjour monGARS', 'Explain recursion', 'Continue'])(
    'keeps plain chat on the streaming path: %s',
    (prompt) => {
      expect(shouldUseNativeAgent(prompt)).toBe(false);
    },
  );
});
