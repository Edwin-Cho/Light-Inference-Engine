/**
 * Layout — 앱 주스켈레톤 (Sidebar + Main Content 영역)
 *
 * 구성:
 *   - 사이드바: LiE 로고, 백엔드 헬스 버지(HealthBadge), 네비게이션, 세션 히스토리
 *   - 메인 콘텐츠: React Router <Outlet /> (페이지별 콘텐츠 렌더링)
 *
 * RBAC: 사용자 role에 따라 네비 메뉴 필터링 (researcher < lab_pi < admin)
 * 세션 rename: 더블클릭 또는 ✏️ 아이콘으로 인라인 수정 활성화
 */
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import type { UserRole } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useSession } from '@/contexts/SessionContext';
import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { MessageSquare, FolderOpen, Settings, LogOut, Wifi, WifiOff, Sun, Moon, Plus, Trash2, Clock, Pencil } from 'lucide-react';
import LiELogo from '@/components/LiELogo';

/** 전체 내비게이션 항목 (minRole 미만은 사이드바에서 숨김) */
const allNavItems = [
  { to: '/', icon: MessageSquare, label: 'Query', minRole: 'researcher' as UserRole },
  { to: '/documents', icon: FolderOpen, label: 'Documents', minRole: 'lab_pi' as UserRole },
  { to: '/admin', icon: Settings, label: 'Admin', minRole: 'admin' as UserRole },
];

/** 역할을 숫자로 매핑 → 단순 비교로 RBAC 접근 제어 */
const ROLE_LEVEL: Record<UserRole, number> = { researcher: 1, lab_pi: 2, admin: 3 };

function hasAccess(userRole: UserRole | null, minRole: UserRole): boolean {
  if (!userRole) return false;
  return ROLE_LEVEL[userRole] >= ROLE_LEVEL[minRole];
}

/** 백엔드 API URL — VITE_API_URL env 미설정 시 개발 서버 기본값 사용 */
const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/**
 * 백엔드 서버 생존 여부를 30초마다 폴링하여 표시하는 배지 컴포넌트
 * null(초기) → pulse 애니메이션, true → 초록 Online, false → 빨간 Offline
 */
function HealthBadge() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        // Bug 1 fix: /api/health → API_BASE/health (프록시 의존 제거, 프로덕션 호환)
        // Improvement 1 fix: AbortController 폴백 (AbortSignal.timeout Chrome103/Safari16 미만 미지원)
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 3000);
        const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(timer);
        setOnline(res.ok);
      } catch {
        setOnline(false);
      }
    };
    // Improvement 3 fix: void check() → IIFE (async 의도 명시적 표현)
    (async () => { await check(); })();
    const id = setInterval(check, 30_000);  // 이후 30초 간격
    return () => clearInterval(id);  // 언마운트 시 인터벌 정리
  }, []);

  if (online === null) return <span className="w-2 h-2 bg-slate-600 rounded-full animate-pulse" />;
  return online ? (
    <span className="flex items-center gap-1.5 text-xs text-emerald-400">
      <Wifi className="w-3 h-3" /> Online
    </span>
  ) : (
    <span className="flex items-center gap-1.5 text-xs text-red-400">
      <WifiOff className="w-3 h-3" /> Offline
    </span>
  );
}

/** 타임스탬프를 "just now / 3m ago / 2h ago / 1d ago" 형태로 변환 */
function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function Layout() {
  const { logout, role } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const { sessions, currentId, startNew, openSession, removeSession, renameSession, clearAllSessions } = useSession();
  // 인라인 이름 수정 상태: editingId가 null이 아니면 해당 세션이 편집 모드
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  const handleLogout = () => {
    clearAllSessions();  // Flaw 3 fix: 로그아웃 시 세션 상태 정리 → 다음 사용자에게 노출 방지
    logout();
    navigate('/login');
  };

  const handleNewChat = () => {
    startNew();
    navigate('/');
  };

  const handleOpenSession = (id: string) => {
    openSession(id);
    navigate('/');
  };

  // Flaw 2 fix: 60초마다 강제 re-render → stale relative time 방지
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 60_000);
    return () => clearInterval(id);
  }, []);

  // Bug 2 fix: click/dblclick 구분 250ms 디바운스 타이머
  const clickTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSessionClick = (id: string) => {
    if (clickTimer.current) clearTimeout(clickTimer.current);
    clickTimer.current = setTimeout(() => {
      if (editingId !== id) handleOpenSession(id);
    }, 250);
  };

  const handleSessionDoubleClick = (e: MouseEvent, id: string, title: string) => {
    e.stopPropagation();
    if (clickTimer.current) clearTimeout(clickTimer.current);
    setEditingId(id);
    setEditingTitle(title);
  };

  // Flaw 1 fix: 빈 문자열 rename 방지 — trim 후 비어있으면 저장 안 함
  const commitRename = (id: string, title: string) => {
    if (title.trim()) renameSession(id, title.trim());
    setEditingId(null);
  };

  return (
    <div className="flex h-screen text-white overflow-hidden" style={{ background: 'var(--layout-bg)' }}>
      {/* Sidebar */}
      <aside className="w-60 flex flex-col shrink-0 border-r"
        style={{ background: 'var(--sidebar-bg)', borderColor: 'var(--sidebar-border)' }}>

        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: 'var(--sidebar-border)' }}>
          <LiELogo />
          <div className="mt-0.5"><HealthBadge /></div>
        </div>

        <nav className="px-3 py-4 space-y-0.5 shrink-0">
          {allNavItems.filter(({ minRole }) => hasAccess(role, minRole)).map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-slate-800 text-slate-100 font-medium'
                    : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/70'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : ''}`} />
                  {label}
                  {isActive && <span className="ml-auto w-1.5 h-1.5 bg-emerald-400 rounded-full" />}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Session History */}
        <div className="flex-1 flex flex-col min-h-0 px-3 border-t" style={{ borderColor: 'var(--sidebar-border)' }}>
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 w-full px-3 py-2.5 mt-3 mb-2 rounded-xl text-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800/70 transition-all duration-150 border border-dashed border-slate-700/60 hover:border-slate-600"
          >
            <Plus className="w-3.5 h-3.5" />
            New Chat
          </button>

          <div className="flex items-center gap-1.5 px-1 mb-1.5">
            <Clock className="w-3 h-3 text-slate-700" />
            <span className="text-[10px] text-slate-700 font-medium uppercase tracking-wider">Recent</span>
          </div>

          {/* 세션 목록: 스크롤 가능, 각 세션은 클릭으로 열기 / 호버 시 편집·삭제 버튼 표시 */}
          <div className="overflow-y-auto flex-1 space-y-0.5 pb-2">
            {sessions.length === 0 ? (
              <p className="text-xs text-slate-700 px-2 py-1.5">No sessions yet</p>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => handleSessionClick(s.id)}
                  className={`group flex items-start gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-150 ${
                    currentId === s.id
                      ? 'bg-slate-800 text-slate-200'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    {editingId === s.id ? (
                      // 편집 모드: 인라인 input — blur 또는 Enter로 저장, Escape로 취소
                      <input
                        autoFocus
                        className="w-full text-xs bg-slate-700 text-slate-100 rounded px-1.5 py-0.5 outline-none border border-emerald-500/50"
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onBlur={() => commitRename(s.id, editingTitle)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            commitRename(s.id, editingTitle);
                          } else if (e.key === 'Escape') {
                            setEditingId(null);  // 변경 없이 취소
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}  // 클릭이 세션 열기로 전파 방지
                      />
                    ) : (
                      // 표시 모드: 더블클릭으로 편집 활성화
                      <p
                        className="text-xs truncate leading-tight"
                        onDoubleClick={(e) => handleSessionDoubleClick(e, s.id, s.title)}
                      >
                        {s.title}
                      </p>
                    )}
                    <p className="text-[10px] text-slate-700 mt-0.5">{formatRelativeTime(s.createdAt)}</p>
                  </div>
                  <div className="flex items-center gap-0.5 shrink-0 mt-0.5">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (clickTimer.current) clearTimeout(clickTimer.current);
                        setEditingId(s.id);
                        setEditingTitle(s.title);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-600 hover:text-emerald-400 transition"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (editingId === s.id) setEditingId(null);  // Flaw 4 fix: 고아 편집 상태 정리
                        removeSession(s.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-slate-600 hover:text-red-400 transition"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="px-3 py-4 border-t space-y-0.5" style={{ borderColor: 'var(--sidebar-border)' }}>
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm text-slate-500 hover:text-slate-300 hover:bg-slate-800/70 transition-all duration-150"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm text-slate-600 hover:text-red-400 hover:bg-red-500/8 transition-all duration-150"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
