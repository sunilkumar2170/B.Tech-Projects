import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ForensicLogo } from '../ui/ForensicLogo';
import { Moon, Sun } from 'lucide-react';
import { api } from '../../services/api';

export const Navbar: React.FC = () => {
  const [isDark, setIsDark] = useState(true);
  const [healthStatus, setHealthStatus] = useState<any>(null);

  useEffect(() => {
    // Check initial dark mode
    if (document.documentElement.classList.contains('light')) {
      setIsDark(false);
    } else {
      document.documentElement.classList.add('dark');
    }

    // Fetch backend health
    api.getHealth()
      .then((data) => setHealthStatus(data))
      .catch(() => setHealthStatus({ status: 'offline' }));
  }, []);

  const toggleTheme = () => {
    if (isDark) {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
      setIsDark(false);
    } else {
      document.documentElement.classList.remove('light');
      document.documentElement.classList.add('dark');
      setIsDark(true);
    }
  };

  const getStatusLabel = () => {
    if (!healthStatus || healthStatus.status === 'offline') {
      return 'Engine Offline';
    }
    if (healthStatus.remote_gpu_worker?.configured) {
      return 'Remote GPU Worker';
    }
    if (healthStatus.checkpoint_status?.checkpoint_present) {
      return 'DFD-FCG Live';
    }
    return 'DFD-FCG Benchmark';
  };

  const isHealthy = healthStatus && healthStatus.status === 'healthy';

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand identity */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
          <div className="w-9 h-9 rounded-xl bg-slate-900 border border-cyan-500/30 flex items-center justify-center p-1 shadow-inner">
            <ForensicLogo size={24} scanning={false} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base sm:text-lg text-slate-100 tracking-tight">
                Video Deepfake Detection
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                CVPR 2025
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block font-mono">
              DFD-FCG Explainable AI Forensic Architecture
            </p>
          </div>
        </Link>

        {/* Right Actions */}
        <div className="flex items-center gap-3">
          {/* Health indicator badge */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
              }`}
            />
            <span className="text-slate-300 font-mono text-[11px]">
              {getStatusLabel()}
            </span>
          </div>

          {/* Theme switcher */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </header>
  );
};
