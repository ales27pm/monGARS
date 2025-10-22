import React from 'react';
import { render, act, waitFor } from '@testing-library/react-native';
import { useDiagnostics } from '../src/hooks/useDiagnostics';
import type { DiagnosticsHookValue } from '../src/hooks/useDiagnostics';

jest.mock('../src/native/diagnostics', () => ({
  refreshNetworkSnapshot: jest.fn().mockResolvedValue({
    timestamp: '2024-01-01T00:00:00.000Z',
    ssid: 'test-network',
    bssid: '00:11:22:33:44:55',
    ip: '192.168.0.10',
    interfaces: [],
    vpnActive: false,
    cellularActive: false,
  }),
  startCapture: jest.fn().mockResolvedValue({
    captureId: 'cap-1',
    path: '/tmp/capture.pcap',
  }),
  stopCapture: jest.fn().mockResolvedValue({
    captureId: 'cap-1',
    path: '/tmp/capture.pcap',
  }),
  enablePacketTunnel: jest.fn().mockResolvedValue(undefined),
}));

const Diagnostics = require('../src/native/diagnostics');

const HookHarness: React.FC<{ onRender: (value: DiagnosticsHookValue) => void }> = ({
  onRender,
}) => {
  const value: DiagnosticsHookValue = useDiagnostics();
  React.useEffect(() => {
    onRender(value);
  }, [value, onRender]);
  return null;
};

describe('useDiagnostics', () => {
  it('loads a network snapshot on mount and exposes refresh', async () => {
    const renders: DiagnosticsHookValue[] = [];
    render(<HookHarness onRender={(value) => renders.push(value)} />);

    await waitFor(() => {
      expect(renders.at(-1)?.snapshot?.ssid).toBe('test-network');
    });

    await act(async () => {
      await renders.at(-1)?.refresh();
    });

    expect(Diagnostics.refreshNetworkSnapshot).toHaveBeenCalledTimes(2);
  });

  it('starts and stops a packet capture while updating state', async () => {
    let latest: DiagnosticsHookValue | null = null;
    render(<HookHarness onRender={(value) => (latest = value)} />);

    await waitFor(() => {
      expect(latest).not.toBeNull();
    });

    await act(async () => {
      const status = await latest!.startCapture({ interfaceName: 'en0', durationSeconds: 5 });
      expect(status.captureId).toBe('cap-1');
    });

    const state = latest!;
    expect(state.loading).toBe(false);
    expect(Diagnostics.startCapture).toHaveBeenCalledWith({ interfaceName: 'en0', durationSeconds: 5 });

    await act(async () => {
      const stopped = await latest!.stopCapture();
      expect(stopped?.captureId).toBe('cap-1');
    });

    expect(Diagnostics.stopCapture).toHaveBeenCalledWith('cap-1');
  });
});
