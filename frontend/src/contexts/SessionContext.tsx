/**
 * SessionContext — 대화 세션 상태 관리 (localStorage 기반)
 *
 * 역할:
 *   - 사이드바 세션 목록(sessions) 전역 공유
 *   - 현재 활성 세션(currentId) 추적
 *   - 메시지 저장(persistMessages), 새 채팅(startNew), 세션 열기/삭제/이름변경
 *
 * 저장소: localStorage (lib/sessions.ts의 CRUD 함수 사용)
 * 소비: Layout.tsx(사이드바), Chat 페이지(메시지 저장)
 */
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import type { ChatMessage, Session } from '@/lib/types';
import { readSessions, upsertSession, deleteSessionById, renameSession as renameSessionStorage, clearAllSessions as clearAllSessionsStorage } from '@/lib/sessions';

/** Context 소비자에게 노출되는 API */
interface SessionContextType {
  sessions: Session[];                                    // localStorage에서 읽은 세션 목록
  currentId: string | null;                              // 현재 사이드바에서 선택된 세션 ID
  sessionKey: number;                                    // 새 채팅 시 Chat 컴포넌트 강제 리마운트용 키
  persistMessages: (messages: ChatMessage[]) => void;    // 대화 내용 저장/갱신
  startNew: () => void;                                  // 새 채팅 시작 (currentId 초기화)
  openSession: (id: string) => void;                     // 기존 세션 불러오기
  removeSession: (id: string) => void;                   // 세션 삭제
  renameSession: (id: string, title: string) => void;    // 세션 이름 변경
  clearAllSessions: () => void;                          // 로그아웃 시 세션 전체 삭제 (localStorage + 상태)
}

const Ctx = createContext<SessionContextType | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  // Improvement 2 fix: corrupted localStorage JSON 파싱 실패 시 SessionProvider 크래시 방지
  const [sessions, setSessions] = useState<Session[]>(() => {
    try {
      return readSessions();
    } catch {
      clearAllSessionsStorage();  // 손상된 데이터 제거 후 비어있는 배열로 복구
      return [];
    }
  });
  const [currentId, setCurrentId] = useState<string | null>(null);   // UI 하이라이트용
  const [sessionKey, setSessionKey] = useState(0);                   // 채팅 컴포넌트 리셋 트리거
  // activeRef: 렌더 사이클과 무관하게 현재 세션 ID를 유지하는 ref
  // useState 대신 ref 사용 이유: 메시지 저장 콜백에서 stale closure 방지
  const activeRef = useRef<string | null>(null);

  // Improvement 3 fix: sessionsRef — persistMessages에서 최신 sessions를 stale closure 없이 읽기
  const sessionsRef = useRef<Session[]>(sessions);
  useEffect(() => { sessionsRef.current = sessions; }, [sessions]);

  const persistMessages = useCallback((messages: ChatMessage[]) => {
    if (messages.length === 0) return;
    const firstUser = messages.find((m) => m.role === 'user');
    if (!firstUser) return;

    // Bug 2 fix: 신규 세션 생성 시 currentId 동기화 → 사이드바 하이라이트 보장
    if (!activeRef.current) {
      // Improvement 1 fix: crypto.randomUUID() → 밀리초 단위 ID 충돌 위험 제거
      const newId = `s_${crypto.randomUUID()}`;
      activeRef.current = newId;
      setCurrentId(newId);
    }

    // Improvement 3: sessionsRef.current으로 stale closure 없이 기존 세션 조회
    const existingSession = sessionsRef.current.find((s) => s.id === activeRef.current);

    // Flaw 2 fix: rename 후 저장 시 title 롤백 방지 — 기존 title 우선 사용
    const raw = firstUser.content;
    const autoTitle = raw.length > 50 ? raw.slice(0, 47) + '…' : raw;
    const title = existingSession?.title ?? autoTitle;

    const session: Session = {
      id: activeRef.current,
      title,
      // Bug 1 fix: 기존 세션의 createdAt 보존 → 매 저장마다 "just now" 갱신 방지
      createdAt: existingSession?.createdAt ?? Date.now(),
      messages,
    };
    // upsertSession: ID가 같으면 덮어쓰기, 없으면 맨 앞에 추가
    const updated = upsertSession(session);
    setSessions(updated);
  }, []);

  const startNew = useCallback(() => {
    activeRef.current = null;      // 다음 메시지 저장 시 새 ID 발급
    setCurrentId(null);            // 사이드바 하이라이트 해제
    setSessionKey((k) => k + 1);   // Chat 컴포넌트를 key 변경으로 완전 리마운트
  }, []);

  const openSession = useCallback((id: string) => {
    activeRef.current = id;          // 이후 저장 시 동일 세션에 덮어쓰도록
    setCurrentId(id);                // 사이드바 하이라이트 변경
    setSessionKey((k) => k + 1);     // Flaw 3 fix: Chat 리마운트 → 이전 세션 입력창/로딩 상태 초기화
  }, []);

  const renameSession = useCallback((id: string, title: string) => {
    // localStorage 업데이트 후 React 상태 동기화
    const updated = renameSessionStorage(id, title);
    setSessions(updated);
  }, []);

  const clearAllSessions = useCallback(() => {
    clearAllSessionsStorage();     // localStorage 전체 삭제
    setSessions([]);               // React 상태 초기화
    activeRef.current = null;      // 저장 기준 세션 ID 초기화
    setCurrentId(null);            // 사이드바 하이라이트 해제
    setSessionKey((k) => k + 1);  // Chat 컴포넌트 리마운트
  }, []);

  const removeSession = useCallback((id: string) => {
    const updated = deleteSessionById(id);
    setSessions(updated);
    if (activeRef.current === id) {
      // Flaw 4 fix: 삭제 후 다음 세션 자동 선택 (없으면 새 채팅 상태)
      const next = updated[0] ?? null;
      activeRef.current = next?.id ?? null;
      setCurrentId(next?.id ?? null);
      setSessionKey((k) => k + 1);
    }
  }, []);

  return (
    <Ctx.Provider
      value={{ sessions, currentId, sessionKey, persistMessages, startNew, openSession, removeSession, renameSession, clearAllSessions }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useSession(): SessionContextType {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
}
