import React from 'react';
import { Role } from '../../types';
import { Shield, Swords, Flame, Target, HeartHandshake } from 'lucide-react';

interface RoleIconProps {
  role: Role;
  className?: string;
}

export const RoleIcon: React.FC<RoleIconProps> = ({ role, className = 'w-4 h-4' }) => {
  switch (role) {
    case 'TOP':
      return <Shield className={`${className} text-amber-400`} title="Top Lane (Tank / Fighter)" />;
    case 'JUNGLE':
      return <Flame className={`${className} text-emerald-400`} title="Jungle (Objective Control)" />;
    case 'MID':
      return <Swords className={`${className} text-cyan-400`} title="Mid Lane (Mage / Assassin)" />;
    case 'ADC':
      return <Target className={`${className} text-rose-400`} title="Bot Lane (Marksman Carry)" />;
    case 'SUPPORT':
      return <HeartHandshake className={`${className} text-indigo-400`} title="Support (Vision / Utility)" />;
    default:
      return null;
  }
};
