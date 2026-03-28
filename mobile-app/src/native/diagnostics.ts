import { TurboModuleRegistry } from 'react-native';
import type { TurboModule } from 'react-native';

export type NetworkInterface = {
  name: string;
  address: string | null;
  mac: string | null;
  isUp: boolean;
};

export type NetworkSnapshot = {
  timestamp: string;
  ssid: string | null;
  bssid: string | null;
  ip: string | null;
  interfaces: NetworkInterface[];
  vpnActive: boolean;
  cellularActive: boolean;
};

export type CaptureOptions = {
  interfaceName: string;
  durationSeconds?: number;
  remoteRviHost?: string;
  outputPath?: string;
};

export type PacketCaptureStatus = {
  captureId: string;
  path: string;
};

export interface DiagnosticsSpec extends TurboModule {
  prepare(): Promise<void>;
  refreshNetworkSnapshot(): Promise<NetworkSnapshot>;
  listInterfaces(): Promise<NetworkInterface[]>;
  startCapture(options: CaptureOptions): Promise<PacketCaptureStatus>;
  stopCapture(captureId: string): Promise<PacketCaptureStatus>;
  enablePacketTunnel(configuration: {
    serverAddress: string;
    username?: string;
    password?: string;
    psk?: string;
  }): Promise<void>;
}

function unavailableError() {
  return new Error('DiagnosticsModule is unavailable in this native build.');
}

const nativeDiagnosticsModule =
  TurboModuleRegistry.get<DiagnosticsSpec>('DiagnosticsModule');

export const diagnosticsModuleAvailable = nativeDiagnosticsModule != null;

const Diagnostics: DiagnosticsSpec =
  nativeDiagnosticsModule ??
  ({
    async prepare() {
      throw unavailableError();
    },
    async refreshNetworkSnapshot() {
      throw unavailableError();
    },
    async listInterfaces() {
      throw unavailableError();
    },
    async startCapture() {
      throw unavailableError();
    },
    async stopCapture() {
      throw unavailableError();
    },
    async enablePacketTunnel() {
      throw unavailableError();
    },
  } as DiagnosticsSpec);

export default Diagnostics;
