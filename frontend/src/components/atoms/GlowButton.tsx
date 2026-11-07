import React from 'react';

interface GlowButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'cyan' | 'pink' | 'amber' | 'slate';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

export const GlowButton: React.FC<GlowButtonProps> = ({
  children,
  variant: _variant = 'slate',
  size = 'md',
  icon,
  className = '',
  disabled,
  type = 'button',
  ...props
}) => {
  const sizeStyles = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base px-5 py-3',
  };

  return (
    <button
      type={type}
      disabled={disabled}
      className={`tool-button ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {icon && <span className="shrink-0" aria-hidden="true">{icon}</span>}
      {children}
    </button>
  );
};
