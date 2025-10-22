import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import ChatScreen from '../screens/ChatScreen';
import DiagnosticsScreen from '../screens/DiagnosticsScreen';
import SettingsScreen from '../screens/SettingsScreen';
import { Platform } from 'react-native';

export type RootStackParamList = {
  Chat: undefined;
  Diagnostics: undefined;
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const RootNavigator = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="Chat"
        component={ChatScreen}
        options={{
          headerLargeTitle: Platform.OS === 'ios',
          title: 'MonGARS',
        }}
      />
      <Stack.Screen
        name="Diagnostics"
        component={DiagnosticsScreen}
        options={{ title: 'Network Diagnostics' }}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ title: 'Settings' }}
      />
    </Stack.Navigator>
  );
};

export default RootNavigator;
