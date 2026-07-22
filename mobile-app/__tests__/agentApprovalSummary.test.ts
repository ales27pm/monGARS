import { summarizeAgentApprovalArguments } from '../src/agent/approvalSummary';

describe('agent approval argument summary', () => {
  it('shows the exact trigger cancellation selector', () => {
    expect(
      summarizeAgentApprovalArguments({
        id: '11111111-2222-4333-8444-555555555555',
      }),
    ).toEqual([{ key: 'id', value: '11111111-2222-4333-8444-555555555555' }]);
    expect(
      summarizeAgentApprovalArguments({ title: 'Morning summary' }),
    ).toEqual([{ key: 'title', value: 'Morning summary' }]);
  });

  it('shows the exact target, subject, title, and timing before approval', () => {
    expect(
      summarizeAgentApprovalArguments({
        body: 'Bonjour',
        startsInMinutes: 30,
        title: 'Rencontre produit',
        subject: 'Décision',
        to: 'alice@example.com',
      }),
    ).toEqual([
      { key: 'to', value: 'alice@example.com' },
      { key: 'title', value: 'Rencontre produit' },
      { key: 'subject', value: 'Décision' },
      { key: 'startsInMinutes', value: '30' },
      { key: 'body', value: 'Bonjour' },
    ]);
  });

  it('shows complete long bodies and redacts credential-shaped fields recursively', () => {
    const summary = summarizeAgentApprovalArguments({
      body: 'x'.repeat(500),
      metadata: { accessToken: 'never-display', safe: 'visible' },
      password: 'never-display',
    });

    expect(summary.find((item) => item.key === 'body')?.value).toBe(
      'x'.repeat(500),
    );
    expect(summary).toContainEqual({ key: 'password', value: '[masqué]' });
    expect(summary.find((item) => item.key === 'metadata')?.value).toBe(
      '{"accessToken":"[masqué]","safe":"visible"}',
    );
  });
});
