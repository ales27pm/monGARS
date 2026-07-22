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
const iosInfoPlistPath = path.join(
  projectRoot,
  'ios',
  'MonGARSMobile',
  'Info.plist',
);
const iosInfoPlistText = existsSync(iosInfoPlistPath)
  ? readFileSync(iosInfoPlistPath, 'utf8')
  : '';
const iosAgentPermissionProviderPath = path.join(
  projectRoot,
  'ios',
  'AgentTools',
  'Sources',
  'MonGARSAgentTools',
  'IOSAgentPermissionProvider.swift',
);
const iosAgentPermissionProviderText = existsSync(
  iosAgentPermissionProviderPath,
)
  ? readFileSync(iosAgentPermissionProviderPath, 'utf8')
  : '';
const iosCoreMLBridgeSourcePath = path.join(
  projectRoot,
  'ios',
  'CoreMLInference',
  'CoreMLInferenceModule.swift',
);
const iosCoreMLBridgeSourceText = existsSync(iosCoreMLBridgeSourcePath)
  ? readFileSync(iosCoreMLBridgeSourcePath, 'utf8')
  : '';
const iosCoreMLExternBridgePath = path.join(
  projectRoot,
  'ios',
  'CoreMLInference',
  'CoreMLInferenceModuleBridge.m',
);
const iosCoreMLExternBridgeText = existsSync(iosCoreMLExternBridgePath)
  ? readFileSync(iosCoreMLExternBridgePath, 'utf8')
  : '';
const iosAppIntentsSourcePath = path.join(
  projectRoot,
  'ios',
  'AppIntents',
  'MonGARSAppIntents.swift',
);
const iosAppIntentsSourceText = existsSync(iosAppIntentsSourcePath)
  ? readFileSync(iosAppIntentsSourcePath, 'utf8')
  : '';
const iosAppShortcutsSourcePath = path.join(
  projectRoot,
  'ios',
  'AppIntents',
  'MonGARSAppShortcuts.swift',
);
const iosAppShortcutsSourceText = existsSync(iosAppShortcutsSourcePath)
  ? readFileSync(iosAppShortcutsSourcePath, 'utf8')
  : '';
const iosAppIntentStorePath = path.join(
  projectRoot,
  'ios',
  'AgentTools',
  'Sources',
  'MonGARSAgentTools',
  'AppIntentHandoffStore.swift',
);
const iosAppIntentStoreText = existsSync(iosAppIntentStorePath)
  ? readFileSync(iosAppIntentStorePath, 'utf8')
  : '';
const appIntentFacadePath = path.join(
  projectRoot,
  'src',
  'native',
  'appIntents.ts',
);
const appIntentFacadeText = existsSync(appIntentFacadePath)
  ? readFileSync(appIntentFacadePath, 'utf8')
  : '';
const appIntentHandoffPath = path.join(
  projectRoot,
  'src',
  'agent',
  'appIntentHandoff.ts',
);
const appIntentHandoffText = existsSync(appIntentHandoffPath)
  ? readFileSync(appIntentHandoffPath, 'utf8')
  : '';
const chatStorePath = path.join(projectRoot, 'src', 'store', 'chatStore.ts');
const chatStoreText = existsSync(chatStorePath)
  ? readFileSync(chatStorePath, 'utf8')
  : '';
const outlookFacadePath = path.join(projectRoot, 'src', 'native', 'outlook.ts');
const outlookFacadeText = existsSync(outlookFacadePath)
  ? readFileSync(outlookFacadePath, 'utf8')
  : '';
const iosEntitlementsPath = path.join(
  projectRoot,
  'ios',
  'MonGARSMobile',
  'MonGARSMobile.entitlements',
);
const iosEntitlementsText = existsSync(iosEntitlementsPath)
  ? readFileSync(iosEntitlementsPath, 'utf8')
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

function textBetween(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start + startMarker.length);
  return start >= 0 && end > start ? source.slice(start, end) : '';
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

function applicationTargetBody(targetName) {
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
  const targetBody = applicationTargetBody(iosApplicationTargetName);
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

function applicationBuildConfigurations() {
  const body = applicationTargetBody(iosApplicationTargetName);
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

function sdkSelectorMatches(selectors, sdkFamily) {
  const sdkSelectors = [
    ...selectors.matchAll(/\[\s*sdk\s*=\s*([^\]]+)\]/gi),
  ].map(([, selector]) => selector.trim().toLowerCase());
  if (sdkSelectors.length === 0) {
    return true;
  }

  return sdkSelectors.every((selector) => {
    if (selector.startsWith(sdkFamily)) {
      return true;
    }
    const pattern = `^${escapeRegExp(selector).replace(/\\\*/g, '.*')}$`;
    const matcher = new RegExp(pattern, 'i');
    return matcher.test(sdkFamily) || matcher.test(`${sdkFamily}18.0`);
  });
}

function deploymentTargetAssignment(text, sdkFamily = 'iphoneos') {
  const assignments = [
    ...text.matchAll(
      /"?IPHONEOS_DEPLOYMENT_TARGET((?:\[[^\]]+\])*)"?\s*=\s*([^;\n]+);?/g,
    ),
  ];
  let resolved = null;
  for (const [, selectors, value] of assignments) {
    if (!sdkSelectorMatches(selectors, sdkFamily)) {
      continue;
    }
    const parsed = numericDeploymentTarget(value);
    if (parsed !== null) {
      resolved = parsed;
    }
  }
  return resolved;
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
        '-sdk',
        'iphoneos',
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

// MonGARSMobile is the only target that links MonGARSCoreML. Unit-test and
// packet-tunnel targets have independent deployment contracts.
function applicationDeploymentTargetsAreAtLeast(minimumMajorVersion) {
  const targetConfigurations = applicationBuildConfigurations();
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
    label: 'Local MonGARSAgentTools package',
    path: 'ios/AgentTools/Package.swift',
    required: true,
    advice:
      'Restore the local Swift package that implements the canonical iOS host tools.',
  },
  {
    label: 'Foreground monGARS App Intents',
    path: 'ios/AppIntents/MonGARSAppIntents.swift',
    required: true,
    advice:
      'Restore the bounded foreground App Intents that stage requests without headless model or tool execution.',
  },
  {
    label: 'monGARS App Shortcuts provider',
    path: 'ios/AppIntents/MonGARSAppShortcuts.swift',
    required: true,
    advice:
      'Restore the AppShortcutsProvider so Siri, Spotlight, and Shortcuts can discover the safe intent surface.',
  },
  {
    label: 'App Intent React Native facade',
    path: 'src/native/appIntents.ts',
    required: true,
    advice:
      'Restore the typed foreground handoff facade; App Intent payloads must never fall back to network execution.',
  },
  {
    label: 'Structured agent React Native facade',
    path: 'src/native/agent.ts',
    required: true,
    advice:
      'Restore the typed agent bridge so approval, permission, and trigger results remain fail-closed.',
  },
  {
    label: 'Outlook React Native facade',
    path: 'src/native/outlook.ts',
    required: true,
    advice:
      'Restore the typed owner-scoped Outlook bridge before enabling Microsoft Graph tools.',
  },
  {
    label: 'Outlook runtime client-ID bridge',
    custom: () =>
      iosCoreMLBridgeSourceText.includes(
        '@objc func configureOutlookClientID(',
      ) &&
      iosCoreMLExternBridgeText.includes(
        'RCT_EXTERN_METHOD(configureOutlookClientID:',
      ) &&
      outlookFacadeText.includes('configureOutlookClientID(') &&
      outlookFacadeText.includes('configureNativeOutlookClientId'),
    required: true,
    advice:
      'Keep the validated public Microsoft client-ID Settings fallback wired across Swift, Objective-C, and TypeScript.',
  },
  {
    label: 'AlarmKit Live Activity source',
    path: 'ios/MonGARSAlarmWidget/MonGARSAlarmWidgetBundle.swift',
    required: true,
    advice:
      'Restore the AlarmKit widget source used by countdown and snooze presentation.',
  },
  {
    label: 'AlarmKit Live Activity Info.plist',
    path: 'ios/MonGARSAlarmWidget/Info.plist',
    required: true,
    advice:
      'Restore the embedded WidgetKit extension metadata for AlarmKit presentation.',
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
    label: 'iOS project registers agent host tools',
    custom: () =>
      applicationBuildPhaseContains('PBXFrameworksBuildPhase', 'Frameworks', [
        'MonGARSAgentTools in Frameworks',
      ]) &&
      iosProjectText.includes(
        'XCLocalSwiftPackageReference "MonGARSAgentTools"',
      ) &&
      iosProjectText.includes('relativePath = AgentTools;'),
    required: true,
    advice:
      'Add the MonGARSAgentTools local package product to the application target.',
  },
  {
    label: 'iOS project registers foreground App Intents',
    custom: () =>
      applicationBuildPhaseContains('PBXSourcesBuildPhase', 'Sources', [
        'MonGARSAppIntents.swift in Sources',
        'MonGARSAppShortcuts.swift in Sources',
      ]) &&
      iosCoreMLBridgeSourceText.includes(
        '@objc func getPendingAppIntentHandoff(',
      ) &&
      iosCoreMLBridgeSourceText.includes(
        '@objc func acknowledgeAppIntentHandoff(',
      ) &&
      iosCoreMLExternBridgeText.includes(
        'RCT_EXTERN_METHOD(getPendingAppIntentHandoff:',
      ) &&
      iosCoreMLExternBridgeText.includes(
        'RCT_EXTERN_METHOD(acknowledgeAppIntentHandoff:',
      ) &&
      iosEntitlementsText.includes('group.com.mongars.mobile'),
    required: true,
    advice:
      'Compile both App Intent sources in MonGARSMobile and keep their protected app-group handoff linked through the native bridge.',
  },
  {
    label: 'iOS app packages its privacy manifest',
    custom: () =>
      existsSync(
        path.join(projectRoot, 'ios', 'MonGARSMobile', 'PrivacyInfo.xcprivacy'),
      ) &&
      applicationBuildPhaseContains('PBXResourcesBuildPhase', 'Resources', [
        'PrivacyInfo.xcprivacy in Resources',
      ]),
    required: true,
    advice:
      'Add PrivacyInfo.xcprivacy to the MonGARSMobile Resources build phase so archives include it.',
  },
  {
    label: 'iOS app entitlement file reference resolves canonically',
    custom: () =>
      iosProjectText.includes(
        'path = MonGARSMobile/MonGARSMobile.entitlements;',
      ),
    required: true,
    advice:
      'Point the MonGARSMobile entitlement PBX file reference at MonGARSMobile/MonGARSMobile.entitlements.',
  },
  {
    label:
      'App Intent profile binding is explicit and read-only lookup is masked',
    custom: () => {
      const pendingLookup = textBetween(
        iosCoreMLBridgeSourceText,
        'func getPendingAppIntentHandoff(',
        'func acknowledgeAppIntentHandoff(',
      );
      return (
        iosCoreMLBridgeSourceText.includes(
          '@objc func setActiveAppIntentProfile(',
        ) &&
        iosCoreMLExternBridgeText.includes(
          'RCT_EXTERN_METHOD(setActiveAppIntentProfile:',
        ) &&
        pendingLookup.includes('profileMatches') &&
        pendingLookup.includes('? handoff.kind.rawValue : "masked"') &&
        !pendingLookup.includes('setActiveProfile(') &&
        appIntentFacadeText.includes('profileMatches: boolean') &&
        appIntentFacadeText.includes("| 'masked'") &&
        appIntentFacadeText.includes(
          'signal ne doit révéler ni kind ni input',
        ) &&
        iosAppIntentStoreText.includes('kind: .masked') &&
        iosAppIntentStoreText.includes('record.kind != .masked') &&
        !iosAppIntentsSourceText.includes('"kind": record.kind.rawValue') &&
        appIntentHandoffText.includes('Action liée à un autre profil') &&
        chatStoreText.includes('setActiveNativeAppIntentProfile(') &&
        chatStoreText.includes('!pending.profileMatches') &&
        chatStoreText.includes('discardNativeAppIntentHandoff(')
      );
    },
    required: true,
    advice:
      'Bind the active owner explicitly during app/session initialization; pending reads must only compare the captured opaque profile and mask mismatched content.',
  },
  {
    label: 'Memory App Intents use one-shot exact native tools without a model',
    custom: () =>
      iosAppIntentStoreText.includes('consumeExactMemoryAction(') &&
      iosCoreMLBridgeSourceText.includes(
        '@objc func executeAppIntentMemoryAction(',
      ) &&
      iosCoreMLBridgeSourceText.includes('? "memory.recall" : "memory.save"') &&
      iosCoreMLBridgeSourceText.includes('"kind": .string("fact")') &&
      iosCoreMLBridgeSourceText.includes(
        'app_intent_memory_add_commit_uncertain',
      ) &&
      iosCoreMLExternBridgeText.includes(
        'RCT_EXTERN_METHOD(executeAppIntentMemoryAction:',
      ) &&
      appIntentFacadeText.includes('executeNativeAppIntentMemoryAction') &&
      chatStoreText.includes('executeNativeAppIntentMemoryAction({') &&
      appIntentHandoffText.includes(
        'Memory App Intents never enter a language-model prompt.',
      ) &&
      !appIntentHandoffText.includes('Use only memory.'),
    required: true,
    advice:
      'Keep memory search/add outside the language model: atomically consume the exact owner-bound protected record, derive only memory.recall or memory.save arguments natively, and never auto-retry the one-shot mutation.',
  },
  {
    label: 'Stored-trigger App Intents bind preview to execution',
    custom: () =>
      chatStoreText.includes('resolvedTrigger') &&
      chatStoreText.includes('previewedTrigger.id') &&
      chatStoreText.includes(
        'sameResolvedTrigger(previewedTrigger, currentTrigger)',
      ) &&
      chatStoreText.includes(
        'Object.assign(reservation, appIntentAgentScope',
      ) &&
      appIntentHandoffText.includes('Requête exacte :'),
    required: true,
    advice:
      'Resolve and display the exact owner-scoped trigger prompt before confirmation, re-resolve its UUID, compare the full snapshot, and bind the deterministic tool scope before acknowledgement.',
  },
  {
    label: 'iOS App Intents require immediate foreground execution',
    custom: () => {
      const intentCount = [
        ...iosAppIntentsSourceText.matchAll(
          /struct\s+MonGARS\w+Intent:\s+AppIntent\s*\{/g,
        ),
      ].length;
      const legacyForegroundCount = [
        ...iosAppIntentsSourceText.matchAll(
          /static\s+let\s+openAppWhenRun\s*=\s*true/g,
        ),
      ].length;
      const ios26ForegroundCount = [
        ...iosAppIntentsSourceText.matchAll(
          /static\s+var\s+supportedModes:\s*IntentModes\s*\{\s*\[\.foreground\(\.immediate\)\]\s*\}/g,
        ),
      ].length;
      return (
        intentCount === 5 &&
        legacyForegroundCount === intentCount &&
        ios26ForegroundCount === intentCount &&
        !iosAppIntentsSourceText.includes(
          'static var supportedModes: IntentModes = .foreground',
        )
      );
    },
    required: true,
    advice:
      'Every App Intent must open monGARS immediately: keep openAppWhenRun for current deployment SDKs and foreground(.immediate) for iOS 26 builds.',
  },
  {
    label: 'iOS App Intents stage one bounded handoff only',
    custom: () =>
      iosAppIntentsSourceText.includes('MonGARSAppIntentHandoffStore.shared') &&
      iosAppIntentsSourceText.includes('store.enqueue(') &&
      [
        'IOSAgentToolExecutor',
        'AgentExecutor',
        'URLSession',
        'runAgent(',
        '.execute(',
      ].every((forbidden) => !iosAppIntentsSourceText.includes(forbidden)),
    required: true,
    advice:
      'App Intent perform methods may only stage the protected foreground handoff; never run models, tools, or network requests there.',
  },
  {
    label: 'iOS App Shortcuts expose the five safe intents',
    custom: () =>
      iosAppShortcutsSourceText.includes(
        'struct MonGARSAppShortcuts: AppShortcutsProvider',
      ) &&
      [
        'MonGARSAskIntent()',
        'MonGARSSearchMemoryIntent()',
        'MonGARSAddMemoryIntent()',
        'MonGARSRunTriggerIntent()',
        'MonGARSDiagnosticsIntent()',
      ].every((intent) => iosAppShortcutsSourceText.includes(intent)) &&
      [...iosAppShortcutsSourceText.matchAll(/AppShortcut\(/g)].length === 5 &&
      [...iosAppShortcutsSourceText.matchAll(/\\\(\.applicationName\)/g)]
        .length >= 5,
    required: true,
    advice:
      'Keep exactly five discoverable shortcuts—ask, local-memory search/add, stored trigger, and passive diagnostics—and include the app name in phrases.',
  },
  {
    label: 'iOS project embeds AlarmKit Live Activity extension',
    custom: () =>
      iosProjectText.includes('MonGARSAlarmWidgetBundle.swift in Sources') &&
      iosProjectText.includes(
        'MonGARSAlarmWidget.appex in Embed App Extensions',
      ) &&
      iosProjectText.includes(
        'productType = "com.apple.product-type.app-extension";',
      ),
    required: true,
    advice:
      'Register and embed the MonGARSAlarmWidget target in the application target.',
  },
  {
    label: 'iOS agent permission usage descriptions',
    custom: () =>
      [
        'NSAlarmKitUsageDescription',
        'NSCalendarsFullAccessUsageDescription',
        'NSCameraUsageDescription',
        'NSContactsUsageDescription',
        'NSHealthShareUsageDescription',
        'NSLocationWhenInUseUsageDescription',
        'NSMotionUsageDescription',
        'NSPhotoLibraryUsageDescription',
        'NSRemindersFullAccessUsageDescription',
      ].every((key) => iosInfoPlistText.includes(`<key>${key}</key>`)),
    required: true,
    advice:
      'Declare every Apple-framework permission used by the canonical agent tools in Info.plist.',
  },
  {
    label: 'iOS EventKit request usage descriptions',
    custom: () =>
      (!iosAgentPermissionProviderText.includes(
        'requestFullAccessToEvents()',
      ) ||
        iosInfoPlistText.includes(
          '<key>NSCalendarsFullAccessUsageDescription</key>',
        )) &&
      (!iosAgentPermissionProviderText.includes(
        'requestWriteOnlyAccessToEvents()',
      ) ||
        iosInfoPlistText.includes(
          '<key>NSCalendarsWriteOnlyAccessUsageDescription</key>',
        )),
    required: true,
    advice:
      'Keep the EventKit permission request APIs and their exact Info.plist usage-description keys in sync.',
  },
  {
    label: 'iOS Microsoft OAuth Info.plist wiring',
    custom: () =>
      iosInfoPlistText.includes('<key>MONGARSMicrosoftClientID</key>') &&
      iosInfoPlistText.includes(
        '<string>$(MONGARS_MICROSOFT_CLIENT_ID)</string>',
      ) &&
      iosInfoPlistText.includes('<key>CFBundleURLTypes</key>') &&
      iosInfoPlistText.includes(
        '<string>msauth.$(PRODUCT_BUNDLE_IDENTIFIER)</string>',
      ),
    required: true,
    advice:
      'Wire the public Microsoft client ID build setting and bundle-derived msauth callback scheme in the app Info.plist.',
  },
  {
    label: 'iOS app build configurations declare Microsoft client ID',
    custom: () => {
      const configurations = applicationBuildConfigurations();
      return (
        configurations.length > 0 &&
        configurations.every(({ body }) =>
          body.includes('MONGARS_MICROSOFT_CLIENT_ID ='),
        )
      );
    },
    required: true,
    advice:
      'Declare MONGARS_MICROSOFT_CLIENT_ID in every app build configuration. The checked-in value may stay empty; override the target setting locally or pass it to xcodebuild.',
  },
  {
    label: 'MonGARSMobile iOS 18 deployment target for stateful Core ML',
    custom: () => applicationDeploymentTargetsAreAtLeast(18),
    required: true,
    advice:
      'The MonGARSMobile app target that links the pinned stateful ML Program requires iOS 18 or newer.',
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
    label: 'iOS agent managed capabilities',
    custom: () =>
      ['com.apple.developer.healthkit', 'com.apple.developer.weatherkit'].every(
        (key) => iosEntitlementsText.includes(`<key>${key}</key>`),
      ),
    required: true,
    advice:
      'Enable HealthKit and WeatherKit for the App ID and keep the signed provisioning profile aligned with the entitlements file.',
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
