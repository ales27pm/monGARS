import { nowISO } from "../utils/time.js";

function makeMessageId() {
  return `msg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createTimelineStore() {
  const order = [];
  const map = new Map();

  function register({
    id,
    role,
    text = "",
    timestamp = nowISO(),
    row,
    metadata = {},
  }) {
    const messageId = id || makeMessageId();
    if (!map.has(messageId)) {
      order.push(messageId);
    }
    map.set(messageId, {
      id: messageId,
      role,
      text,
      timestamp,
      row,
      metadata: { ...metadata },
    });
    if (row) {
      row.dataset.messageId = messageId;
      row.dataset.role = role;
      row.dataset.rawText = text;
      row.dataset.timestamp = timestamp;
    }
    return messageId;
  }

  function update(id, patch = {}) {
    if (!map.has(id)) {
      return null;
    }
    const entry = map.get(id);
    const next = { ...entry, ...patch };
    if (patch && typeof patch.metadata === "object" && patch.metadata !== null) {
      const merged = { ...entry.metadata };
      Object.entries(patch.metadata).forEach(([key, value]) => {
        if (value === undefined || value === null) {
          delete merged[key];
        } else {
          merged[key] = value;
        }
      });
      next.metadata = merged;
    }
    map.set(id, next);
    const { row } = next;
    if (row && row.isConnected) {
      if (next.text !== entry.text) {
        row.dataset.rawText = next.text || "";
      }
      if (next.timestamp !== entry.timestamp) {
        row.dataset.timestamp = next.timestamp || "";
      }
      if (next.role && next.role !== entry.role) {
        row.dataset.role = next.role;
      }
    }
    return next;
  }

  function collect() {
    return order
      .map((id) => {
        const entry = map.get(id);
        if (!entry) {
          return null;
        }
        return {
          role: entry.role,
          text: entry.text,
          timestamp: entry.timestamp,
          ...(entry.metadata &&
            Object.keys(entry.metadata).length > 0 && {
              metadata: { ...entry.metadata },
            }),
        };
      })
      .filter(Boolean);
  }

  function clear() {
    order.length = 0;
    map.clear();
  }

  return {
    register,
    update,
    collect,
    clear,
    order,
    map,
    makeMessageId,
  };
}
