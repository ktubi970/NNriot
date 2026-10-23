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
  variant = 'slate',
  size = 'sm',
  pulse = false,
  className = '',
}) => {
  const color = variant === 'cyan' ? 'team-blue' : variant === 'pink' ? 'team-red' : 'text-slate-300';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border border-[#414752] bg-[#272a31] font-medium ${color} ${size === 'sm' ? 'text-[11px] px-2 py-0.5' : 'text-xs px-2.5 py-1'} ${className}`}>
      {pulse && <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />}
      {children}
    </span>
  );
};
