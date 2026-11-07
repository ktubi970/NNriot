import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Champion, Role } from '../../types';
import { Search, X, Shield, Flame, Swords, Target, HeartHandshake } from 'lucide-react';
import { CyberBadge } from '../atoms/CyberBadge';

interface ChampionPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (champion: Champion) => void;
  targetRole: Role;
  champions: Champion[];
  currentChampionId?: string;
}

export const ChampionPickerModal: React.FC<ChampionPickerModalProps> = ({
  isOpen,
  onClose,
  onSelect,
  targetRole,
  champions,
  currentChampionId,
}) => {
  const [search, setSearch] = useState('');
  const [selectedRole, setSelectedRole] = useState<Role | 'ALL'>(targetRole);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setSelectedRole(targetRole);
      setSearch('');
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen, targetRole]);

  const filteredChampions = useMemo(() => {
    return champions.filter(c => {
      const matchesSearch = c.name.toLowerCase().includes(search.toLowerCase());
      const matchesRole = selectedRole === 'ALL' || c.roles.includes(selectedRole);
      return matchesSearch && matchesRole;
    });
  }, [champions, search, selectedRole]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-3xl bg-[#0b1120] border border-cyan-500/40 rounded-xl shadow-[0_0_50px_rgba(0,240,255,0.15)] overflow-hidden flex flex-col max-h-[85vh]">
        {/* Top Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#070b14]">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-cyan-400 animate-ping" />
            <h2 className="text-lg font-mono font-bold tracking-wider text-cyan-400 uppercase">
              SELECT CHAMPION [{targetRole}]
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filters */}
        <div className="p-4 border-b border-slate-800/80 bg-[#090e1a]/90 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              ref={inputRef}
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search champion by name..."
              className="w-full pl-9 pr-4 py-2 bg-slate-900/90 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 font-mono"
            />
          </div>

          <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
            {(['ALL', 'TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'] as const).map(role => (
              <button
                key={role}
                onClick={() => setSelectedRole(role)}
                className={`px-2.5 py-1 text-xs font-mono rounded flex items-center gap-1 transition-all ${
                  selectedRole === role
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {role === 'TOP' && <Shield className="w-3 h-3 text-amber-400" />}
                {role === 'JUNGLE' && <Flame className="w-3 h-3 text-emerald-400" />}
                {role === 'MID' && <Swords className="w-3 h-3 text-cyan-400" />}
                {role === 'ADC' && <Target className="w-3 h-3 text-rose-400" />}
                {role === 'SUPPORT' && <HeartHandshake className="w-3 h-3 text-indigo-400" />}
                {role}
              </button>
            ))}
          </div>
        </div>

        {/* Champions Grid */}
        <div className="p-6 overflow-y-auto grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-3 cyber-grid">
          {filteredChampions.map(c => {
            const isCurrent = c.id === currentChampionId;
            return (
              <button
                key={c.id}
                onClick={() => {
                  onSelect(c);
                  onClose();
                }}
                className={`group relative flex flex-col items-center p-2 rounded-lg border transition-all duration-150 ${
                  isCurrent
                    ? 'bg-cyan-950/40 border-cyan-400 ring-1 ring-cyan-400'
                    : 'bg-slate-900/70 border-slate-800 hover:border-cyan-500/60 hover:bg-slate-800/80 hover:shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                }`}
              >
                <div className="relative w-14 h-14 rounded-md overflow-hidden border border-slate-700 group-hover:border-cyan-400/80 transition-colors">
                  <img
                    src={c.iconUrl}
                    alt={c.name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/Aatrox.png';
                    }}
                  />
                  {c.winRate && (
                    <div className="absolute bottom-0 inset-x-0 bg-black/80 text-[9px] font-mono text-center text-emerald-400 py-0.5">
                      {c.winRate.toFixed(1)}%
                    </div>
                  )}
                </div>

                <span className="mt-1.5 text-xs font-medium text-slate-300 group-hover:text-cyan-300 truncate max-w-full font-mono">
                  {c.name}
                </span>

                <div className="flex gap-0.5 mt-1">
                  {c.roles.map(r => (
                    <span key={r} className="text-[8px] px-1 py-0.2 bg-slate-950 text-slate-400 rounded">
                      {r.slice(0, 1)}
                    </span>
                  ))}
                </div>
              </button>
            );
          })}

          {filteredChampions.length === 0 && (
            <div className="col-span-full py-12 text-center text-slate-500 font-mono text-sm">
              NO CHAMPIONS FOUND FOR "{search}"
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#070b14] flex items-center justify-between text-xs text-slate-400 font-mono">
          <span>{filteredChampions.length} champions matches</span>
          <CyberBadge variant="slate" size="sm">PRESS ESC TO CLOSE</CyberBadge>
        </div>
      </div>
    </div>
  );
};
