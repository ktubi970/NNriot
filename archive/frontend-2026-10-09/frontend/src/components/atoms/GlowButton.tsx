import React from 'react';

interface GlowButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'cyan' | 'pink' | 'amber' | 'slate';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const GlowButton: React.FC<GlowButtonProps> = ({
  children,
  variant = 'cyan',
  size = 'md',
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const variantStyles = {
    cyan: 'bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border-cyan-500/50 hover:border-cyan-400 hover:shadow-[0_0_15px_rgba(0,240,255,0.4)] active:scale-[0.98]',
    pink: 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border-rose-500/50 hover:border-rose-400 hover:shadow-[0_0_15px_rgba(244,63,94,0.4)] active:scale-[0.98]',
    amber: 'bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border-amber-500/50 hover:border-amber-400 hover:shadow-[0_0_15px_rgba(245,158,11,0.4)] active:scale-[0.98]',
    slate: 'bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 border-slate-700 hover:border-slate-500 hover:text-white',
  };

  const sizeStyles = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5 font-mono',
    md: 'text-sm px-3.5 py-2 gap-2 font-mono',
    lg: 'text-base px-5 py-2.5 gap-2.5 font-medium',
  };

  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded border font-mono transition-all duration-200 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none ${
        variantStyles[variant]
      } ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </button>
  );
};
