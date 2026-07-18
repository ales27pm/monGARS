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
const iosXcodeProjectPath = path.join(
  projectRoot,
  'ios',
  'MonGARSMobile.xcodeproj',
);
const iosProjectPath = path.join(iosXcodeProjectPath, 'project.pbxproj');
const iosProjectText = existsSync(iosProjectPath)
  ? readFileSync(iosProjectPath, 'utf8')
  : '';
const hostHasXcodebuild =
  spawnSync('xcodebuild', ['-version'], { stdio: 'ignore' }).status === 0;
const iosApplicationTargetName = 'MonGARSMobile';

function pbxSection(name) {
  const start = `/* Begin ${name} section */`;
  const end = `/* End ${name} section */`;
  const startIndex = iosProjectText.indexOf(start);
  const endIndex = iosProjectText.indexOf(end);
  if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
    return '';
  }
  return iosProjectText.slice(startIndex + start.length, endIndex);
}

function pbxObjectBody(section, objectID) {
  const marker = `${objectID} /*`;
  const markerIndex = section.indexOf(marker);
  if (markerIndex === -1) {
    return '';
  }
  const objectStart = section.indexOf('{', markerIndex);
  if (objectStart === -1) {
    return '';
  }
  const lineEnd = section.indexOf('\n', objectStart);
  const inlineObjectEnd = section.indexOf('};', objectStart);
  const objectEnd =
    inlineObjectEnd !== -1 && (lineEnd === -1 || inlineObjectEnd < lineEnd)
      ? inlineObjectEnd
      : section.indexOf('\n\t\t};', objectStart);
  if (objectEnd === -1) {
    return '';
  }
  return section.slice(objectStart + 1, objectEnd);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function pbxObjectID(section, comment) {
  const match = section.match(
    new RegExp(`([A-F0-9]{24}) \\/\\* ${escapeRegExp(comment)} \\*\\/ = \\{`),
  );
  return match?.[1] ?? null;
}

function pbxProperty(body, property) {
  const match = body.match(new RegExp(`\\b${property}\\s*=\\s*([^;]+);`));
  return match?.[1]?.trim().replace(/^"|"$/g, '') ?? null;
}

function nativeTargetBody(targetName) {
  const targets = pbxSection('PBXNativeTarget');
  const targetID = pbxObjectID(targets, targetName);
  if (!targetID) {
    return '';
  }
  const body = pbxObjectBody(targets, targetID);
  if (
    pbxProperty(body, 'name') !== targetName ||
    pbxProperty(body, 'productType') !== 'com.apple.product-type.application'
  ) {
    return '';
  }
  return body;
}

function applicationTargetBuildPhaseID(phaseName) {
  const targetBody = nativeTargetBody(iosApplicationTargetName);
  if (!targetBody) {
    return null;
  }
  const phaseMatch = targetBody.match(
    new RegExp(`([A-F0-9]{24}) \\/\\* ${phaseName} \\*\\/`),
  );
  return phaseMatch?.[1] ?? null;
}

function applicationBuildPhaseContains(phaseType, phaseName, entries) {
  const phaseID = applicationTargetBuildPhaseID(phaseName);
  if (!phaseID) {
    return false;
  }
  const phaseBody = pbxObjectBody(pbxSection(phaseType), phaseID);
  return entries.every((entry) => phaseBody.includes(entry));
}

function buildConfigurationsForList(configurationListID) {
  const listBody = pbxObjectBody(
    pbxSection('XCConfigurationList'),
    configurationListID,
  );
  const configurationIDs = [
    ...listBody.matchAll(/([A-F0-9]{24}) \/\* [^\n]+ \*\//g),
  ].map(([, objectID]) => objectID);
  const configurations = pbxSection('XCBuildConfiguration');
  return configurationIDs
    .map((objectID) => pbxObjectBody(configurations, objectID))
    .filter(Boolean)
    .map((body) => ({ body, name: pbxProperty(body, 'name') }));
}

function targetBuildConfigurations(targetName) {
  const body = nativeTargetBody(targetName);
  const listID = body.match(/buildConfigurationList\s*=\s*([A-F0-9]{24})/)?.[1];
  return listID ? buildConfigurationsForList(listID) : [];
}

function projectBuildConfigurations() {
  const projects = pbxSection('PBXProject');
  const projectID = projects.match(
    /\n\t\t([A-F0-9]{24}) \/\* Project object \*\/ = \{/,
  )?.[1];
  const body = projectID ? pbxObjectBody(projects, projectID) : '';
  const listID = body.match(/buildConfigurationList\s*=\s*([A-F0-9]{24})/)?.[1];
  return listID ? buildConfigurationsForList(listID) : [];
}

function numericDeploymentTarget(value) {
  const normalized = value
    ?.replace(/\$\(inherited\)/g, '')
    .trim()
    .replace(/^"|"$/g, '');
  return normalized && /^\d+(?:\.\d+)*$/.test(normalized)
    ? Number(normalized)
    : null;
}

function deploymentTargetAssignment(text) {
  const assignments = [
    ...text.matchAll(
      /IPHONEOS_DEPLOYMENT_TARGET(?:\[[^\]]+\])?\s*=\s*([^;\n]+);?/g,
    ),
  ];
  for (const [, value] of assignments.reverse()) {
    const parsed = numericDeploymentTarget(value);
    if (parsed !== null) {
      return parsed;
    }
  }
  return null;
}

function resolveFileReference(referenceID) {
  const body = pbxObjectBody(pbxSection('PBXFileReference'), referenceID);
  const configuredPath = pbxProperty(body, 'path') ?? pbxProperty(body, 'name');
  if (!configuredPath) {
    return null;
  }

  const iosRoot = path.join(projectRoot, 'ios');
  const candidates = [
    path.resolve(iosRoot, configuredPath),
    path.resolve(projectRoot, configuredPath),
  ];
  const direct = candidates.find((candidate) => existsSync(candidate));
  if (direct) {
    return direct;
  }

  const discovered = findFile(
    iosRoot,
    (_, name) => name === path.basename(configuredPath),
  );
  return discovered ? path.join(projectRoot, discovered) : null;
}

function deploymentTargetFromXcconfig(filePath, includeStack = new Set()) {
  if (!filePath || includeStack.has(filePath) || !existsSync(filePath)) {
    return null;
  }
  const nextIncludeStack = new Set(includeStack);
  nextIncludeStack.add(filePath);

  let resolved = null;
  const lines = readFileSync(filePath, 'utf8').split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.replace(/\/\/.*$/, '').trim();
    const include = line.match(/^#include\??\s+["<]([^">]+)[">]/);
    if (include) {
      const includedValue = deploymentTargetFromXcconfig(
        path.resolve(path.dirname(filePath), include[1]),
        nextIncludeStack,
      );
      if (includedValue !== null) {
        resolved = includedValue;
      }
      continue;
    }
    const assignment = deploymentTargetAssignment(line);
    if (assignment !== null) {
      resolved = assignment;
    }
  }
  return resolved;
}

function deploymentTargetFromConfiguration(configuration) {
  const direct = deploymentTargetAssignment(configuration.body);
  if (direct !== null) {
    return direct;
  }
  const referenceID = configuration.body.match(
    /baseConfigurationReference\s*=\s*([A-F0-9]{24})/,
  )?.[1];
  return referenceID
    ? deploymentTargetFromXcconfig(resolveFileReference(referenceID))
    : null;
}

function xcodebuildDeploymentTargets(configurations) {
  if (!hostHasXcodebuild) {
    return null;
  }
  const values = [];
  for (const configuration of configurations) {
    const result = spawnSync(
      'xcodebuild',
      [
        '-project',
        iosXcodeProjectPath,
        '-target',
        iosApplicationTargetName,
        '-configuration',
        configuration.name,
        '-disableAutomaticPackageResolution',
        '-showBuildSettings',
      ],
      { encoding: 'utf8', timeout: 30_000 },
    );
    if (result.status !== 0) {
      return null;
    }
    const match = result.stdout.match(
      /^\s*IPHONEOS_DEPLOYMENT_TARGET\s*=\s*(\S+)\s*$/m,
    );
    const parsed = numericDeploymentTarget(match?.[1]);
    if (parsed === null) {
      return null;
    }
    values.push(parsed);
  }
  return values;
}

function allDeploymentTargetsAreAtLeast(minimumMajorVersion) {
  const targetConfigurations = targetBuildConfigurations(
    iosApplicationTargetName,
  );
  if (targetConfigurations.length === 0) {
    return false;
  }

  const effectiveValues = xcodebuildDeploymentTargets(targetConfigurations);
  if (effectiveValues) {
    return effectiveValues.every((value) => value >= minimumMajorVersion);
  }

  const projectConfigurations = new Map(
    projectBuildConfigurations().map((configuration) => [
      configuration.name,
      deploymentTargetFromConfiguration(configuration),
    ]),
  );
  const staticValues = targetConfigurations.map(
    (configuration) =>
      deploymentTargetFromConfiguration(configuration) ??
      projectConfigurations.get(configuration.name) ??
      null,
  );
  return staticValues.every(
    (value) => value !== null && value >= minimumMajorVersion,
  );
}

const checks = [
  {
    label: 'JavaScript entrypoint',
    path: 'index.js',
    required: true,
    advice:
      'Restore the React Native entry files before attempting a native build.',
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
    advice:
      'Voice input will be unavailable on iOS until the native module is restored.',
  },
  {
    label: 'Diagnostics native module',
    path: 'ios/Diagnostics/DiagnosticsModule.swift',
    required: false,
    advice:
      'Packet capture and tunnel diagnostics are unavailable on iOS until the native module is restored.',
  },
  {
    label: 'Core ML inference native module',
    path: 'ios/CoreMLInference/CoreMLInferenceModule.swift',
    required: true,
    advice:
      'The on-device backend needs its Swift bridge before it can load or run the model.',
  },
  {
    label: 'Core ML React Native facade',
    path: 'src/native/coreml.ts',
    required: true,
    advice:
      'Restore the typed JavaScript facade so native failures remain explicit and events can be normalized.',
  },
  {
    label: 'Core ML inference state store',
    path: 'src/store/inferenceStore.ts',
    required: true,
    advice:
      'Restore the inference state store that serializes generation, cancellation, and model lifecycle operations.',
  },
  {
    label: 'Core ML inference Objective-C bridge',
    path: 'ios/CoreMLInference/CoreMLInferenceModuleBridge.m',
    required: true,
    advice:
      'React Native cannot expose the Swift Core ML module without its extern bridge.',
  },
  {
    label: 'Local MonGARSCoreML package',
    path: 'ios/MonGARSCoreML/Package.swift',
    required: true,
    advice:
      'Restore the local Swift package that owns model download, tokenization, and generation.',
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
    label: 'iOS project registers Core ML inference',
    custom: () =>
      applicationBuildPhaseContains('PBXSourcesBuildPhase', 'Sources', [
        'CoreMLInferenceModule.swift in Sources',
        'CoreMLInferenceModuleBridge.m in Sources',
      ]) &&
      applicationBuildPhaseContains('PBXFrameworksBuildPhase', 'Frameworks', [
        'MonGARSCoreML in Frameworks',
      ]) &&
      iosProjectText.includes('XCLocalSwiftPackageReference "MonGARSCoreML"') &&
      iosProjectText.includes('relativePath = MonGARSCoreML;'),
    required: true,
    advice:
      'Add the Core ML bridge and MonGARSCoreML package product to the app target.',
  },
  {
    label: 'iOS 18 deployment target for stateful Core ML',
    custom: () => allDeploymentTargetsAreAtLeast(18),
    required: true,
    advice: 'The pinned stateful ML Program requires iOS 18 or newer.',
  },
  {
    label: 'iOS frameworks use the active SDK',
    custom: () =>
      !iosProjectText.includes('Platforms/iPhoneOS.platform/Developer/SDKs/'),
    required: true,
    advice:
      'Replace versioned SDK paths with SDKROOT so new Xcode releases can resolve frameworks.',
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
      findFile(
        path.join(projectRoot, 'android', 'app', 'src', 'main', 'java'),
        (_, name) => /Voice.*\.(kt|java)$/.test(name),
      ),
    required: false,
    advice:
      'Android can build now, but native voice support is still missing and will be feature-gated.',
  },
  {
    label: 'Android native diagnostics module',
    matcher: () =>
      findFile(
        path.join(projectRoot, 'android', 'app', 'src', 'main', 'java'),
        (_, name) => /Diagnostics.*\.(kt|java)$/.test(name),
      ),
    required: false,
    advice:
      'Android can build now, but diagnostics support is still missing and will be feature-gated.',
  },
  {
    label: 'Env key MONGARS_BASE_URL',
    custom: () => envKeys.has('MONGARS_BASE_URL'),
    required: true,
    advice:
      'Define MONGARS_BASE_URL in .env.example for a single source of truth.',
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
    advice:
      'Define MONGARS_VOICE_LOCALE in .env.example for voice recognition defaults.',
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
  console.log(
    `${marker} ${check.label}${ok && found !== check.label ? ` -> ${found}` : ''}`,
  );

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
