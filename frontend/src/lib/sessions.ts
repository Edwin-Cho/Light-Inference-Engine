import type { Session } from './types';

const STORAGE_KEY = 'lie_sessions';
const MAX_SESSIONS = 50;

export function readSessions(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Session[]) : [];
  } catch {
    return [];
  }
}

function writeSessions(sessions: Session[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch {
    /* quota exceeded — silent */
  }
}

export function upsertSession(session: Session): Session[] {
  const all = readSessions();
  const idx = all.findIndex((s) => s.id === session.id);
  let updated: Session[];
  if (idx >= 0) {
    updated = all.map((s, i) => (i === idx ? { ...session, createdAt: s.createdAt } : s));
  } else {
    updated = [session, ...all].slice(0, MAX_SESSIONS);
  }
  writeSessions(updated);
  return updated;
}

export function deleteSessionById(id: string): Session[] {
  const updated = readSessions().filter((s) => s.id !== id);
  writeSessions(updated);
  return updated;
}

export function renameSession(id: string, title: string): Session[] {
  const updated = readSessions().map((s) =>
    s.id === id ? { ...s, title: title.trim() || s.title } : s,
  );
  writeSessions(updated);
  return updated;
}

export function clearAllSessions(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
