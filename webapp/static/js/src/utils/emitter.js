export function createEmitter() {
  const listeners = new Map();

  function on(event, handler) {
    if (!listeners.has(event)) {
      listeners.set(event, new Set());
    }
    listeners.get(event).add(handler);
    return () => off(event, handler);
  }

  function off(event, handler) {
    if (!listeners.has(event)) return;
    const bucket = listeners.get(event);
    bucket.delete(handler);
    if (bucket.size === 0) {
      listeners.delete(event);
    }
  }

  function emit(event, payload) {
    if (!listeners.has(event)) return;
    listeners.get(event).forEach((handler) => {
      try {
        handler(payload);
      } catch (err) {
        console.error("Emitter handler error", err);
      }
    });
  }

  return { on, off, emit };
}
