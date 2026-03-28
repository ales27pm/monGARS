import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');

function findFile(root, predicate) {
  if (!existsSync(root)) {
    return null;
  }

  const queue = [root];

  while (queue.length > 0) {
    const current = queue.shift();
    const entries = readdirSync(current, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        queue.push(fullPath);
        continue;
      }
      if (predicate(fullPath, entry.name)) {
        return path.relative(projectRoot, fullPath);
      }
    }
  }

  return null;
}

function readEnvTemplate() {
  const envPath = path.join(projectRoot, '.env.example');
  if (!existsSync(envPath)) {
    return new Set();
  }

  return new Set(
    readFileSync(envPath, 'utf8')
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'))
      .map((line) => line.split('=', 1)[0]),
  );
}

const envKeys = readEnvTemplate();
const iosProjectPath = path.join(projectRoot, 'ios', 'MonGARSMobile.xcodeproj', 'project.pbxproj');
const iosProjectText = existsSync(iosProjectPath)
  ? readFileSync(iosProjectPath, 'utf8')
  : '';
const hostHasXcodebuild = spawnSync('xcodebuild', ['-version'], { stdio: 'ignore' }).status === 0;

const checks = [
  {
    label: 'JavaScript entrypoint',
    path: 'index.js',
    required: true,
    advice: 'Restore the React Native entry files before attempting a native build.',
  },
  {
    label: 'Metro config',
    path: 'metro.config.js',
    required: true,
    advice: 'Metro is required for both iOS and Android bundling.',
  },
  {
    label: 'iOS Podfile',
    path: 'ios/Podfile',
    required: true,
    advice:
      'The iOS app shell is incomplete. Restore or regenerate the native iOS project before running pod-install.',
  },
  {
    label: 'iOS Xcode project',
    custom: () =>
      existsSync(path.join(projectRoot, 'ios', 'MonGARSMobile.xcodeproj')) ||
      findFile(path.join(projectRoot, 'ios'), (_, name) =>
        name.endsWith('.xcworkspace'),
      ),
    required: true,
    advice:
      'No Xcode project or workspace was found under ios/. A build plugin cannot ship an IPA without it.',
  },
  {
    label: 'iOS Info.plist',
    path: 'ios/MonGARSMobile/Info.plist',
    required: true,
    advice:
      'Add an Info.plist so microphone, speech recognition, ATS, and app metadata can be declared.',
  },
  {
    label: 'iOS packet tunnel Info.plist',
    path: 'ios/DiagnosticsExtension/Info.plist',
    required: false,
    advice:
      'The diagnostics extension needs its own Info.plist with the packet-tunnel extension point metadata.',
  },
  {
    label: 'Android app manifest',
    path: 'android/app/src/main/AndroidManifest.xml',
    required: true,
    advice:
      'The Android app shell is also incomplete. Restore android/app before expecting native parity.',
  },
  {
    label: 'Android app module',
    path: 'android/app/build.gradle',
    required: true,
    advice:
      'The Gradle app module is missing. Without it, Android builds and signing are impossible.',
  },
  {
    label: 'Voice native module',
    path: 'ios/Voice/VoiceModule.swift',
    required: false,
    advice: 'Voice input will be unavailable on iOS until the native module is restored.',
  },
  {
    label: 'Diagnostics native module',
    path: 'ios/Diagnostics/DiagnosticsModule.swift',
    required: false,
    advice:
      'Packet capture and tunnel diagnostics are unavailable on iOS until the native module is restored.',
  },
  {
    label: 'iOS packet tunnel provider',
    path: 'ios/DiagnosticsExtension/PacketCaptureProvider.swift',
    required: false,
    advice:
      'Packet capture needs a Network Extension provider target to write shared diagnostic captures.',
  },
  {
    label: 'iOS project registers VoiceModule',
    custom: () => iosProjectText.includes('VoiceModule.swift'),
    required: false,
    advice:
      'The Xcode project exists, but VoiceModule.swift is not part of the app target yet.',
  },
  {
    label: 'iOS project registers DiagnosticsModule',
    custom: () => iosProjectText.includes('DiagnosticsModule.swift'),
    required: false,
    advice:
      'The Xcode project exists, but DiagnosticsModule.swift is not part of the app target yet.',
  },
  {
    label: 'iOS project registers packet tunnel extension',
    custom: () =>
      iosProjectText.includes('PacketCaptureProvider.swift') &&
      iosProjectText.includes('com.apple.product-type.app-extension'),
    required: false,
    advice:
      'The packet tunnel provider exists on disk, but the Xcode project does not embed a Network Extension target yet.',
  },
  {
    label: 'iOS app entitlements',
    path: 'ios/MonGARSMobile/MonGARSMobile.entitlements',
    required: false,
    advice:
      'Add app entitlements for packet-tunnel and shared app-group access before enabling diagnostics on-device.',
  },
  {
    label: 'iOS packet tunnel entitlements',
    path: 'ios/DiagnosticsExtension/PacketCapture.entitlements',
    required: false,
    advice:
      'The packet tunnel extension needs matching entitlements to access the shared capture container.',
  },
  {
    label: 'Host xcodebuild',
    custom: () => hostHasXcodebuild,
    required: false,
    advice:
      'The repo is wired for iOS now, but pod install and IPA archives still require macOS with Xcode installed.',
  },
  {
    label: 'Android native voice module',
    matcher: () =>
      findFile(path.join(projectRoot, 'android', 'app', 'src', 'main', 'java'), (_, name) =>
        /Voice.*\.(kt|java)$/.test(name),
      ),
    required: false,
    advice:
      'Android can build now, but native voice support is still missing and will be feature-gated.',
  },
  {
    label: 'Android native diagnostics module',
    matcher: () =>
      findFile(path.join(projectRoot, 'android', 'app', 'src', 'main', 'java'), (_, name) =>
        /Diagnostics.*\.(kt|java)$/.test(name),
      ),
    required: false,
    advice:
      'Android can build now, but diagnostics support is still missing and will be feature-gated.',
  },
  {
    label: 'Env key MONGARS_BASE_URL',
    custom: () => envKeys.has('MONGARS_BASE_URL'),
    required: true,
    advice: 'Define MONGARS_BASE_URL in .env.example for a single source of truth.',
  },
  {
    label: 'Env key MONGARS_WS_URL',
    custom: () => envKeys.has('MONGARS_WS_URL'),
    required: true,
    advice: 'Define MONGARS_WS_URL in .env.example for realtime ticketed chat.',
  },
  {
    label: 'Env key MONGARS_VOICE_LOCALE',
    custom: () => envKeys.has('MONGARS_VOICE_LOCALE'),
    required: true,
    advice: 'Define MONGARS_VOICE_LOCALE in .env.example for voice recognition defaults.',
  },
];

let failed = false;

console.log('monGARS native preflight\n');

for (const check of checks) {
  let found = null;

  if (typeof check.custom === 'function') {
    found = check.custom() ? check.label : null;
  } else if (typeof check.matcher === 'function') {
    found = check.matcher();
  } else if (check.path) {
    found = existsSync(path.join(projectRoot, check.path)) ? check.path : null;
  }

  const ok = Boolean(found);
  const marker = ok ? '[ok]' : check.required ? '[missing]' : '[warn]';
  console.log(`${marker} ${check.label}${ok && found !== check.label ? ` -> ${found}` : ''}`);

  if (!ok && check.advice) {
    console.log(`      ${check.advice}`);
  }

  if (!ok && check.required) {
    failed = true;
  }
}

if (failed) {
  console.error(
    '\nNative preflight failed. Restore or regenerate the missing iOS/Android shells before attempting store builds.',
  );
  process.exit(1);
}

console.log('\nNative preflight passed.');
