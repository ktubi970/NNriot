import React from 'react';
import { Role } from '../../types';
import { Shield, Swords, Flame, Target, HeartHandshake } from 'lucide-react';

interface RoleIconProps {
  role: Role;
  className?: string;
}

const roleDetails = {
  TOP: { icon: Shield, label: 'Top' },
  JUNGLE: { icon: Flame, label: 'Jungle' },
  MID: { icon: Swords, label: 'Mid' },
  ADC: { icon: Target, label: 'ADC' },
  SUPPORT: { icon: HeartHandshake, label: 'Support' },
};

export const RoleIcon: React.FC<RoleIconProps> = ({ role, className = 'w-4 h-4' }) => {
  const detail = roleDetails[role];
  if (!detail) return null;
  const Icon = detail.icon;
  return <span role="img" aria-label={detail.label} title={detail.label} className="inline-flex shrink-0 text-slate-400"><Icon aria-hidden="true" className={className} /></span>;
};
