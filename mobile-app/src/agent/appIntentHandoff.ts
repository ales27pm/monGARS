import type {
  NativeAppIntentHandoff,
  NativeResolvedStoredTrigger,
} from '../native/appIntents';

export const APP_INTENT_MAXIMUM_AGENT_PROMPT_BYTES = 512;

const utf8ByteLength = (value: string): number => {
  let length = 0;
  for (const character of value) {
    const codePoint = character.codePointAt(0) ?? 0;
    length +=
      codePoint <= 0x7f
        ? 1
        : codePoint <= 0x7ff
          ? 2
          : codePoint <= 0xffff
            ? 3
            : 4;
  }
  return length;
};

const boundedAgentPrompt = (value: string | null): string | null => {
  if (!value || utf8ByteLength(value) > APP_INTENT_MAXIMUM_AGENT_PROMPT_BYTES) {
    return null;
  }
  return value;
};

export const appIntentHandoffTitle = (
  handoff: NativeAppIntentHandoff,
): string => {
  if (!handoff.profileMatches || handoff.kind === 'masked') {
    return 'Action liée à un autre profil';
  }
  switch (handoff.kind) {
    case 'ask':
      return 'Question pour monGARS';
    case 'memorySearch':
      return 'Recherche dans la mémoire locale';
    case 'memoryAdd':
      return 'Ajout à la mémoire locale';
    case 'runTrigger':
      return 'Déclencheur enregistré';
    case 'diagnostics':
      return 'Diagnostics passifs';
  }
};

export const appIntentHandoffPreview = (
  handoff: NativeAppIntentHandoff,
  resolvedTrigger?: NativeResolvedStoredTrigger | null,
): string => {
  if (!handoff.profileMatches || handoff.kind === 'masked') {
    return "Le contenu reste masqué. Activez le profil lié pour l'examiner.";
  }
  if (handoff.kind === 'diagnostics') {
    return 'Ouvrir l’écran de diagnostic. Aucune capture ne démarrera automatiquement.';
  }
  if (handoff.kind === 'runTrigger') {
    if (!resolvedTrigger) {
      return "Aucun déclencheur unique et valide n'a pu être prévisualisé.";
    }
    return [
      `Déclencheur : ${resolvedTrigger.title}`,
      `Requête exacte : ${resolvedTrigger.prompt}`,
      resolvedTrigger.repeats ? 'Récurrent' : 'Ponctuel',
    ].join('\n');
  }
  return handoff.input ?? '';
};

export const appIntentHandoffPrompt = (
  handoff: NativeAppIntentHandoff,
  resolvedTrigger?: NativeResolvedStoredTrigger,
): string | null => {
  if (!handoff.profileMatches || handoff.kind === 'masked') {
    return null;
  }
  const input = handoff.input?.trim();
  switch (handoff.kind) {
    case 'ask':
      return boundedAgentPrompt(input || null);
    case 'memorySearch':
    case 'memoryAdd':
      // Native consumes the protected record and derives exact one-tool args.
      // Memory App Intents never enter a language-model prompt.
      return null;
    case 'runTrigger':
      return boundedAgentPrompt(resolvedTrigger?.prompt.trim() || null);
    case 'diagnostics':
      return null;
  }
};
