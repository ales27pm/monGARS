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

const Diagnostics =
  TurboModuleRegistry.getEnforcing<DiagnosticsSpec>('DiagnosticsModule');

export default Diagnostics;
