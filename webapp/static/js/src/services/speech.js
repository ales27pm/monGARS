import { createEmitter } from "../utils/emitter.js";
import { nowISO } from "../utils/time.js";

function normalizeText(value) {
  if (!value) {
    return "";
  }
  return String(value).replace(/\s+/g, " ").trim();
}

function describeRecognitionError(code, fallback = "") {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return (
        "Accès au microphone refusé. Autorisez la dictée vocale dans votre navigateur."
      );
    case "network":
      return "La reconnaissance vocale a été interrompue par un problème réseau.";
    case "no-speech":
      return "Aucune voix détectée. Essayez de parler plus près du micro.";
    case "aborted":
      return "La dictée a été interrompue.";
    case "audio-capture":
      return "Aucun microphone disponible. Vérifiez votre matériel.";
    case "bad-grammar":
      return "Le service de dictée a rencontré une erreur de traitement.";
    default:
      return fallback || "La reconnaissance vocale a rencontré une erreur inattendue.";
  }
}

function mapVoice(voice) {
  return {
    name: voice.name,
    lang: voice.lang,
    voiceURI: voice.voiceURI,
    default: Boolean(voice.default),
    localService: Boolean(voice.localService),
  };
}

export function createSpeechService({ defaultLanguage } = {}) {
  const emitter = createEmitter();
  const globalScope = typeof window !== "undefined" ? window : {};
  const RecognitionCtor =
    globalScope.SpeechRecognition || globalScope.webkitSpeechRecognition || null;
  const recognitionSupported = Boolean(RecognitionCtor);
  const synthesisSupported = Boolean(globalScope.speechSynthesis);
  const synth = synthesisSupported ? globalScope.speechSynthesis : null;

  let recognition = null;
  const navigatorLanguage =
    typeof navigator !== "undefined" && navigator.language
      ? navigator.language
      : null;
  let recognitionLang =
    defaultLanguage || navigatorLanguage || "fr-CA";
  let manualStop = false;
  let listening = false;
  let speaking = false;
  let preferredVoiceURI = null;
  let voicesCache = [];
  let microphonePrimed = false;
  let microphonePriming = null;

  const userAgent =
    typeof navigator !== "undefined" && navigator.userAgent
      ? navigator.userAgent.toLowerCase()
      : "";
  const isAppleMobile = /iphone|ipad|ipod/.test(userAgent);
  const isSafari =
    /safari/.test(userAgent) &&
    !/crios|fxios|chrome|android|edge|edg|opr|opera/.test(userAgent);

  function requiresMicrophonePriming() {
    if (!recognitionSupported) {
      return false;
    }
    if (microphonePrimed) {
      return false;
    }
    if (typeof navigator === "undefined") {
      return false;
    }
    if (
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      return false;
    }
    return isAppleMobile && isSafari;
  }

  function releaseStream(stream) {
    if (!stream) {
      return;
    }
    const tracks = typeof stream.getTracks === "function" ? stream.getTracks() : [];
    tracks.forEach((track) => {
      try {
        track.stop();
      } catch (err) {
        console.debug("Unable to stop track", err);
      }
    });
  }

  async function ensureMicrophoneAccess() {
    if (!requiresMicrophonePriming()) {
      return true;
    }
    if (microphonePrimed) {
      return true;
    }
    if (microphonePriming) {
      try {
        return await microphonePriming;
      } catch (err) {
        return false;
      }
    }
    microphonePriming = navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        microphonePrimed = true;
        releaseStream(stream);
        return true;
      })
      .catch((err) => {
        emitError({
          source: "recognition",
          code: "permission-denied",
          message:
            "Autorisation du microphone refusée. Activez l'accès dans les réglages de Safari.",
          details: err,
        });
        return false;
      })
      .finally(() => {
        microphonePriming = null;
      });
    return microphonePriming;
  }

  function isPermissionError(error) {
    if (!error) {
      return false;
    }
    const code =
      typeof error === "string"
        ? error
        : error.name || error.code || error.message || "";
    const normalised = String(code).toLowerCase();
    return [
      "notallowederror",
      "not-allowed",
      "service-not-allowed",
      "securityerror",
      "permissiondeniederror",
      "aborterror",
    ].some((candidate) => normalised.includes(candidate));
  }

  function emitError(payload) {
    const enriched = {
      timestamp: nowISO(),
      ...payload,
    };
    console.error("Speech service error", enriched);
    emitter.emit("error", enriched);
  }

  function ensureRecognition() {
    if (!recognitionSupported) {
      return null;
    }
    if (recognition) {
      return recognition;
    }
    recognition = new RecognitionCtor();
    recognition.lang = recognitionLang;
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      listening = true;
      emitter.emit("listening-change", {
        listening: true,
        reason: "start",
        timestamp: nowISO(),
      });
    };

    recognition.onend = () => {
      const reason = manualStop ? "manual" : "ended";
      listening = false;
      emitter.emit("listening-change", {
        listening: false,
        reason,
        timestamp: nowISO(),
      });
      manualStop = false;
    };

    recognition.onerror = (event) => {
      listening = false;
      const code = event.error || "unknown";
      emitError({
        source: "recognition",
        code,
        message: describeRecognitionError(code, event.message),
        event,
      });
      emitter.emit("listening-change", {
        listening: false,
        reason: "error",
        code,
        timestamp: nowISO(),
      });
    };

    recognition.onresult = (event) => {
      if (!event.results) {
        return;
      }
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (!result || result.length === 0) {
          continue;
        }
        const alternative = result[0];
        const transcript = normalizeText(alternative?.transcript || "");
        if (!transcript) {
          continue;
        }
        emitter.emit("transcript", {
          transcript,
          isFinal: Boolean(result.isFinal),
          confidence:
            typeof alternative.confidence === "number"
              ? alternative.confidence
              : null,
          timestamp: nowISO(),
        });
      }
    };

    recognition.onaudioend = () => {
      emitter.emit("audio-end", { timestamp: nowISO() });
    };

    recognition.onspeechend = () => {
      emitter.emit("speech-end", { timestamp: nowISO() });
    };

    return recognition;
  }

  async function startListening(options = {}) {
    if (!recognitionSupported) {
      emitError({
        source: "recognition",
        code: "unsupported",
        message: "La dictée vocale n'est pas disponible sur cet appareil.",
      });
      return false;
    }
    const instance = ensureRecognition();
    if (!instance) {
      return false;
    }
    if (listening) {
      return true;
    }
    manualStop = false;
    recognitionLang = normalizeText(options.language) || recognitionLang;
    instance.lang = recognitionLang;
    instance.interimResults = options.interimResults !== false;
    instance.continuous = Boolean(options.continuous);
    instance.maxAlternatives = options.maxAlternatives || 1;
    if (requiresMicrophonePriming()) {
      const granted = await ensureMicrophoneAccess();
      if (!granted) {
        return false;
      }
    }
    try {
      instance.start();
      return true;
    } catch (err) {
      if (requiresMicrophonePriming() && !microphonePrimed && isPermissionError(err)) {
        const granted = await ensureMicrophoneAccess();
        if (granted) {
          try {
            instance.start();
            return true;
          } catch (retryErr) {
            emitError({
              source: "recognition",
              code: "start-failed",
              message:
                retryErr && retryErr.message
                  ? retryErr.message
                  : "Impossible de démarrer la reconnaissance vocale.",
              details: retryErr,
            });
            return false;
          }
        }
      }
      emitError({
        source: "recognition",
        code: "start-failed",
        message:
          err && err.message
            ? err.message
            : "Impossible de démarrer la reconnaissance vocale.",
        details: err,
      });
      return false;
    }
  }

  function stopListening(options = {}) {
    if (!recognition) {
      return;
    }
    manualStop = true;
    try {
      if (options && options.abort && typeof recognition.abort === "function") {
        recognition.abort();
      } else {
        recognition.stop();
      }
    } catch (err) {
      emitError({
        source: "recognition",
        code: "stop-failed",
        message: "Arrêt de la dictée impossible.",
        details: err,
      });
    }
  }

  function findVoice(uri) {
    if (!uri || !synth) {
      return null;
    }
    const voices = synth.getVoices();
    return voices.find((voice) => voice.voiceURI === uri) || null;
  }

  function refreshVoices() {
    if (!synth) {
      return [];
    }
    try {
      voicesCache = synth.getVoices();
      const payload = voicesCache.map(mapVoice);
      emitter.emit("voices", { voices: payload });
      return payload;
    } catch (err) {
      emitError({
        source: "synthesis",
        code: "voices-failed",
        message: "Impossible de récupérer la liste des voix disponibles.",
        details: err,
      });
      return [];
    }
  }

  function speak(text, options = {}) {
    if (!synthesisSupported) {
      emitError({
        source: "synthesis",
        code: "unsupported",
        message: "La synthèse vocale n'est pas disponible sur cet appareil.",
      });
      return null;
    }
    const content = normalizeText(text);
    if (!content) {
      return null;
    }
    if (listening) {
      stopListening({ abort: true });
    }
    stopSpeaking();
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = normalizeText(options.lang) || recognitionLang;
    const rate = Number(options.rate);
    if (!Number.isNaN(rate) && rate > 0) {
      utterance.rate = Math.min(rate, 2);
    }
    const pitch = Number(options.pitch);
    if (!Number.isNaN(pitch) && pitch > 0) {
      utterance.pitch = Math.min(pitch, 2);
    }
    const voice =
      findVoice(options.voiceURI) || findVoice(preferredVoiceURI) || null;
    if (voice) {
      utterance.voice = voice;
    }

    utterance.onstart = () => {
      speaking = true;
      emitter.emit("speaking-change", {
        speaking: true,
        utterance,
        timestamp: nowISO(),
      });
    };

    utterance.onend = () => {
      speaking = false;
      emitter.emit("speaking-change", {
        speaking: false,
        utterance,
        timestamp: nowISO(),
      });
    };

    utterance.onerror = (event) => {
      speaking = false;
      emitError({
        source: "synthesis",
        code: event.error || "unknown",
        message:
          event && event.message
            ? event.message
            : "La synthèse vocale a rencontré une erreur.",
        event,
      });
      emitter.emit("speaking-change", {
        speaking: false,
        utterance,
        reason: "error",
        timestamp: nowISO(),
      });
    };

    synth.speak(utterance);
    return utterance;
  }

  function stopSpeaking() {
    if (!synthesisSupported) {
      return;
    }
    if (synth.speaking || synth.pending) {
      synth.cancel();
    }
    if (speaking) {
      speaking = false;
      emitter.emit("speaking-change", {
        speaking: false,
        reason: "cancel",
        timestamp: nowISO(),
      });
    }
  }

  function setPreferredVoice(uri) {
    preferredVoiceURI = uri || null;
  }

  function setLanguage(lang) {
    const next = normalizeText(lang);
    if (next) {
      recognitionLang = next;
      if (recognition) {
        recognition.lang = recognitionLang;
      }
    }
  }

  if (synthesisSupported) {
    refreshVoices();
    if (synth.addEventListener) {
      synth.addEventListener("voiceschanged", refreshVoices);
    } else if ("onvoiceschanged" in synth) {
      synth.onvoiceschanged = refreshVoices;
    }
  }

  return {
    on: emitter.on,
    off: emitter.off,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    setPreferredVoice,
    setLanguage,
    refreshVoices,
    getVoices: () => voicesCache.map(mapVoice),
    getPreferredVoice: () => preferredVoiceURI,
    isRecognitionSupported: () => recognitionSupported,
    isSynthesisSupported: () => synthesisSupported,
  };
}
