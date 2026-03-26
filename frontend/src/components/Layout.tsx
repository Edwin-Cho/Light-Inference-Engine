import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import type { UserRole } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useEffect, useState } from 'react';
import { BookOpen, MessageSquare, FolderOpen, Settings, LogOut, Wifi, WifiOff, Sun, Moon } from 'lucide-react';

const allNavItems = [
  { to: '/', icon: MessageSquare, label: 'Query', minRole: 'researcher' as UserRole },
  { to: '/documents', icon: FolderOpen, label: 'Documents', minRole: 'lab_pi' as UserRole },
  { to: '/admin', icon: Settings, label: 'Admin', minRole: 'admin' as UserRole },
];

const ROLE_LEVEL: Record<UserRole, number> = { researcher: 1, lab_pi: 2, admin: 3 };

function hasAccess(userRole: UserRole | null, minRole: UserRole): boolean {
  if (!userRole) return false;
  return ROLE_LEVEL[userRole] >= ROLE_LEVEL[minRole];
}

function HealthBadge() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(3000) });
        setOnline(res.ok);
      } catch {
        setOnline(false);
      }
    };
    void check();
    const id = setInterval(() => void check(), 30000);
    return () => clearInterval(id);
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

export default function Layout() {
  const { logout, role } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen text-white overflow-hidden" style={{ background: 'var(--layout-bg)' }}>
      {/* Sidebar */}
      <aside className="w-60 flex flex-col shrink-0 border-r"
        style={{ background: 'var(--sidebar-bg)', borderColor: 'var(--sidebar-border)' }}>

        <div className="flex items-center gap-3 px-5 py-5 border-b" style={{ borderColor: 'var(--sidebar-border)' }}>
          <div className="relative">
            <div className="absolute inset-0 bg-slate-600/30 rounded-xl blur-sm" />
            <div className="relative w-9 h-9 bg-gradient-to-br from-purple-500 to-purple-700 rounded-xl flex items-center justify-center">
              <BookOpen className="w-4 h-4 text-white" />
            </div>
          </div>
          <div>
            <p className="text-white font-semibold text-sm leading-tight">Scholar RAG</p>
            <div className="mt-0.5"><HealthBadge /></div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {allNavItems.filter(({ minRole }) => hasAccess(role, minRole)).map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-slate-700/60 text-slate-100 font-medium'
                    : 'text-slate-500 hover:text-slate-200 hover:bg-white/5'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : ''}`} />
                  {label}
                  {isActive && <span className="ml-auto w-1.5 h-1.5 bg-cyan-400 rounded-full" />}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="px-3 py-4 border-t space-y-0.5" style={{ borderColor: 'var(--sidebar-border)' }}>
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-xl text-sm text-slate-500 hover:text-slate-300 hover:bg-white/5 transition-all duration-150"
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
