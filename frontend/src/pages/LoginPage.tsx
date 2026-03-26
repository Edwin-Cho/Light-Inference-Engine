import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { BookOpen, Loader2, Lock, User } from 'lucide-react';

export default function LoginPage() {
  const { login } = useAuth();
  const { theme } = useTheme();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(username, password);
      navigate('/');
    } catch {
      setError('Invalid credentials. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const bgStyle = theme === 'dark'
    ? { background: 'radial-gradient(ellipse at 60% 20%, #2d1b6940 0%, transparent 60%), radial-gradient(ellipse at 10% 80%, #1e3a5f30 0%, transparent 50%), #0a0a0f' }
    : { background: 'radial-gradient(ellipse at 60% 20%, #e0e7ff60 0%, transparent 60%), radial-gradient(ellipse at 10% 80%, #dbeafe40 0%, transparent 50%), #f1f5f9' };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden"
      style={bgStyle}>


      <div className="w-full max-w-sm px-4 relative z-10">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="relative mb-5">
            <div className="absolute inset-0 bg-indigo-500/20 rounded-2xl blur-lg" />
            <div className="relative w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-700 rounded-2xl flex items-center justify-center shadow-2xl">
              <BookOpen className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Scholar RAG</h1>
          <p className="text-slate-500 text-sm mt-1">OnDevice · Offline · Private</p>
        </div>

        {/* Card */}
        <div className="bg-white/[0.04] border border-white/10 rounded-2xl p-7 shadow-2xl backdrop-blur-md">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username"
                required
                className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500/60 focus:border-indigo-500/40 transition text-sm"
              />
            </div>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                required
                className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500/60 focus:border-indigo-500/40 transition text-sm"
              />
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-2.5 text-red-400 text-xs flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-red-400 rounded-full shrink-0" />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 text-sm mt-1"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? 'Authenticating…' : 'Sign In'}
            </button>
          </form>
        </div>

        <p className="text-center text-slate-500 text-xs mt-6">
          All queries run locally · No data leaves this device
        </p>
      </div>
    </div>
  );
}
