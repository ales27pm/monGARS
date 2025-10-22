import React, { useMemo, useState } from 'react';
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useDiagnostics } from '../hooks/useDiagnostics';

const DiagnosticsScreen: React.FC = () => {
  const {
    snapshot,
    refresh,
    startCapture,
    stopCapture,
    capture,
    error,
    loading,
  } = useDiagnostics();
  const [duration] = useState(120);

  const interfaces = useMemo(() => snapshot?.interfaces ?? [], [snapshot]);

  const handleStart = async (name: string) => {
    try {
      await startCapture({ interfaceName: name, durationSeconds: duration });
      Alert.alert('Capture démarrée', `Interface ${name}`);
    } catch (err) {
      Alert.alert('Erreur', (err as Error).message);
    }
  };

  const handleStop = async () => {
    if (!capture) {
      return;
    }
    const status = await stopCapture();
    Alert.alert('Capture terminée', status?.path ?? '');
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>État réseau</Text>
        <Pressable
          style={styles.refreshButton}
          onPress={refresh}
          disabled={loading}
        >
          <Text style={styles.refreshText}>
            {loading ? 'Actualisation…' : 'Actualiser'}
          </Text>
        </Pressable>
      </View>
      {error ? <Text style={styles.error}>{error}</Text> : null}
      {snapshot ? (
        <View style={styles.snapshot}>
          <Text style={styles.label}>SSID: {snapshot.ssid ?? '—'}</Text>
          <Text style={styles.label}>Adresse IP: {snapshot.ip ?? '—'}</Text>
          <Text style={styles.label}>
            VPN actif: {snapshot.vpnActive ? 'Oui' : 'Non'}
          </Text>
          <Text style={styles.label}>
            Cellulaire actif: {snapshot.cellularActive ? 'Oui' : 'Non'}
          </Text>
        </View>
      ) : null}
      <Text style={styles.sectionTitle}>Interfaces</Text>
      {interfaces.map((item) => (
        <View key={item.name} style={styles.interfaceRow}>
          <View>
            <Text style={styles.interfaceName}>{item.name}</Text>
            <Text style={styles.interfaceMeta}>
              Adresse: {item.address ?? '—'}
            </Text>
            <Text style={styles.interfaceMeta}>MAC: {item.mac ?? '—'}</Text>
          </View>
          <Pressable
            style={styles.captureButton}
            onPress={() => handleStart(item.name)}
          >
            <Text style={styles.captureText}>Sniffer</Text>
          </Pressable>
        </View>
      ))}
      {capture ? (
        <Pressable style={styles.stopButton} onPress={handleStop}>
          <Text style={styles.stopText}>
            Arrêter la capture ({capture.captureId})
          </Text>
        </Pressable>
      ) : null}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 24,
    gap: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    color: '#0f172a',
  },
  refreshButton: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 10,
  },
  refreshText: {
    color: '#f8fafc',
    fontWeight: '600',
  },
  snapshot: {
    backgroundColor: '#0f172a',
    borderRadius: 12,
    padding: 18,
  },
  label: {
    color: '#f8fafc',
    marginBottom: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  interfaceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#cbd5f5',
  },
  interfaceName: {
    fontWeight: '600',
    color: '#0f172a',
  },
  interfaceMeta: {
    color: '#475569',
  },
  captureButton: {
    backgroundColor: '#0f172a',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
  },
  captureText: {
    color: '#f8fafc',
  },
  stopButton: {
    backgroundColor: '#dc2626',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  stopText: {
    color: '#f8fafc',
    fontWeight: '700',
  },
  error: {
    color: '#ef4444',
  },
});

export default DiagnosticsScreen;
