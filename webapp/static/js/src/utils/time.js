export function nowISO() {
  return new Date().toISOString();
}

export function formatTimestamp(ts) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleString("fr-CA");
  } catch (err) {
    return String(ts);
  }
}
