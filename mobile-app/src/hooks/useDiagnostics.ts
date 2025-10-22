import { useEffect, useState } from 'react';
import Diagnostics, {
  CaptureOptions,
  NetworkSnapshot,
  PacketCaptureStatus,
} from '../native/diagnostics';

export type DiagnosticsHookValue = {
  snapshot: NetworkSnapshot | null;
  capture: PacketCaptureStatus | null;
  error: string | null;
  loading: boolean;
  refresh: () => Promise<void>;
  startCapture: (options: CaptureOptions) => Promise<PacketCaptureStatus>;
  stopCapture: () => Promise<PacketCaptureStatus | null>;
  enablePacketTunnel: (serverAddress: string) => Promise<void>;
};

export function useDiagnostics(): DiagnosticsHookValue {
  const [snapshot, setSnapshot] = useState<NetworkSnapshot | null>(null);
  const [capture, setCapture] = useState<PacketCaptureStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    Diagnostics.refreshNetworkSnapshot()
      .then(setSnapshot)
      .catch((err) => setError(err.message));
  }, []);

  const refresh = async () => {
    try {
      const next = await Diagnostics.refreshNetworkSnapshot();
      setSnapshot(next);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const startCapture = async (options: CaptureOptions) => {
    setLoading(true);
    setError(null);
    try {
      const status = await Diagnostics.startCapture(options);
      setCapture(status);
      setLoading(false);
      return status;
    } catch (err) {
      setLoading(false);
      const message = (err as Error).message;
      setError(message);
      throw err;
    }
  };

  const stopCapture = async () => {
    if (!capture) {
      return null;
    }
    try {
      const status = await Diagnostics.stopCapture(capture.captureId);
      setCapture(null);
      return status;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    }
  };

  const enablePacketTunnel = async (serverAddress: string) => {
    setError(null);
    await Diagnostics.enablePacketTunnel({ serverAddress });
  };

  return {
    snapshot,
    capture,
    error,
    loading,
    refresh,
    startCapture,
    stopCapture,
    enablePacketTunnel,
  };
}
