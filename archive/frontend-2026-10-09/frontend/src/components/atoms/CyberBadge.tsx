import React from 'react';

interface CyberBadgeProps {
  children: React.ReactNode;
  variant?: 'cyan' | 'pink' | 'green' | 'amber' | 'purple' | 'slate';
  size?: 'sm' | 'md';
  pulse?: boolean;
  className?: string;
}

export const CyberBadge: React.FC<CyberBadgeProps> = ({
  children,
  variant = 'cyan',
  size = 'sm',
  pulse = false,
  className = '',
}) => {
  const variantStyles = {
    cyan: 'bg-cyan-950/60 text-cyan-400 border-cyan-500/40 shadow-[0_0_10px_rgba(0,240,255,0.15)]',
    pink: 'bg-rose-950/60 text-rose-400 border-rose-500/40 shadow-[0_0_10px_rgba(244,63,94,0.15)]',
    green: 'bg-emerald-950/60 text-emerald-400 border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.15)]',
    amber: 'bg-amber-950/60 text-amber-400 border-amber-500/40 shadow-[0_0_10px_rgba(245,158,11,0.15)]',
    purple: 'bg-purple-950/60 text-purple-400 border-purple-500/40 shadow-[0_0_10px_rgba(168,85,247,0.15)]',
    slate: 'bg-slate-900/60 text-slate-400 border-slate-700/60',
  };

  const sizeStyles = {
    sm: 'text-[10px] px-2 py-0.5 font-mono tracking-wider',
    md: 'text-xs px-2.5 py-1 font-mono tracking-wide',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border uppercase font-medium backdrop-blur-sm transition-all ${
        variantStyles[variant]
      } ${sizeStyles[size]} ${pulse ? 'animate-pulse' : ''} ${className}`}
    >
      {pulse && (
        <span className={`w-1.5 h-1.5 rounded-full ${
          variant === 'cyan' ? 'bg-cyan-400' :
          variant === 'pink' ? 'bg-rose-400' :
          variant === 'green' ? 'bg-emerald-400' :
          variant === 'amber' ? 'bg-amber-400' : 'bg-slate-400'
        } animate-ping`} />
      )}
      {children}
    </span>
  );
};
