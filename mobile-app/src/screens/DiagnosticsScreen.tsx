import React, { useState } from 'react';
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useDiagnostics } from '../hooks/useDiagnostics';
import { useChatStore } from '../store/chatStore';

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
  const { connection } = useChatStore();
  const [duration] = useState(120);
  const [refreshing, setRefreshing] = useState(false);

  const interfaces = snapshot?.interfaces ?? [];

  const handleStart = async (name: string) => {
    try {
      await startCapture({ interfaceName: name, durationSeconds: duration });
      Alert.alert('Capture demarree', `Interface ${name}`);
    } catch (err) {
      Alert.alert('Erreur', (err as Error).message);
    }
  };

  const handleStop = async () => {
    if (!capture) {
      return;
    }
    try {
      const status = await stopCapture();
      Alert.alert('Capture terminee', status?.path ?? '');
    } catch (err) {
      Alert.alert('Erreur', (err as Error).message);
    }
  };

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.container}>
      <View style={styles.card}>
        <View style={styles.header}>
          <Text style={styles.title}>Temps reel</Text>
          <Pressable
            style={styles.refreshButton}
            onPress={async () => {
              setRefreshing(true);
              try {
                await refresh();
              } catch (err) {
                Alert.alert('Erreur', (err as Error).message);
              } finally {
                setRefreshing(false);
              }
            }}
            disabled={loading || refreshing}
          >
            <Text style={styles.refreshText}>
              {refreshing ? 'Actualisation…' : 'Actualiser'}
            </Text>
          </Pressable>
        </View>
        <Text style={styles.detail}>Etat: {connection.status}</Text>
        <Text style={styles.detail}>Detail: {connection.detail ?? '—'}</Text>
        <Text style={styles.detail}>
          Dernier message:{' '}
          {connection.lastMessageAt
            ? connection.lastMessageAt.toLocaleTimeString()
            : '—'}
        </Text>
        <Text style={styles.detail}>
          Latence: {connection.latencyMs ? `${connection.latencyMs} ms` : '—'}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.title}>Reseau natif</Text>
        {error ? <Text style={styles.error}>{error}</Text> : null}
        {snapshot ? (
          <View style={styles.snapshot}>
            <Text style={styles.snapshotText}>
              SSID: {snapshot.ssid ?? '—'}
            </Text>
            <Text style={styles.snapshotText}>
              Adresse IP: {snapshot.ip ?? '—'}
            </Text>
            <Text style={styles.snapshotText}>
              VPN actif: {snapshot.vpnActive ? 'Oui' : 'Non'}
            </Text>
            <Text style={styles.snapshotText}>
              Cellulaire actif: {snapshot.cellularActive ? 'Oui' : 'Non'}
            </Text>
          </View>
        ) : null}
      </View>

      <View style={styles.card}>
        <Text style={styles.title}>Interfaces</Text>
        {interfaces.map((item) => (
          <View key={item.name} style={styles.interfaceRow}>
            <View style={styles.interfaceCopy}>
              <Text style={styles.interfaceName}>{item.name}</Text>
              <Text style={styles.interfaceMeta}>
                Adresse: {item.address ?? '—'}
              </Text>
              <Text style={styles.interfaceMeta}>MAC: {item.mac ?? '—'}</Text>
            </View>
            <Pressable
              style={styles.captureButton}
              onPress={() => handleStart(item.name)}
              disabled={!!capture || loading}
            >
              <Text style={styles.captureText}>Sniffer</Text>
            </Pressable>
          </View>
        ))}
        {capture ? (
          <Pressable style={styles.stopButton} onPress={handleStop}>
            <Text style={styles.stopText}>
              Arreter la capture ({capture.captureId})
            </Text>
          </Pressable>
        ) : null}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#060b16',
  },
  container: {
    padding: 20,
    gap: 18,
  },
  card: {
    borderRadius: 28,
    padding: 20,
    backgroundColor: '#0d1525',
    borderWidth: 1,
    borderColor: '#1f2d45',
    gap: 14,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: '800',
    color: '#f8fafc',
  },
  refreshButton: {
    backgroundColor: '#f59e0b',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 14,
  },
  refreshText: {
    color: '#111827',
    fontWeight: '800',
  },
  detail: {
    color: '#cbd5e1',
  },
  snapshot: {
    backgroundColor: '#07101d',
    borderRadius: 18,
    padding: 16,
    gap: 6,
  },
  snapshotText: {
    color: '#f8fafc',
  },
  interfaceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 14,
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#30415f',
  },
  interfaceCopy: {
    flex: 1,
    gap: 4,
  },
  interfaceName: {
    color: '#f8fafc',
    fontWeight: '700',
  },
  interfaceMeta: {
    color: '#94a3b8',
  },
  captureButton: {
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#172033',
  },
  captureText: {
    color: '#dbeafe',
    fontWeight: '700',
  },
  stopButton: {
    backgroundColor: '#b91c1c',
    padding: 16,
    borderRadius: 18,
    alignItems: 'center',
  },
  stopText: {
    color: '#fee2e2',
    fontWeight: '800',
  },
  error: {
    color: '#fca5a5',
  },
});

export default DiagnosticsScreen;
