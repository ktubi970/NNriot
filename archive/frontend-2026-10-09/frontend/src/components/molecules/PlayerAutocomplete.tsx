import React, { useState, useEffect, useRef } from 'react';
import { Role, PlayerSearchResult } from '../../types';
import { searchPlayers, getRoleProSuggestions } from '../../services/api';
import { User, Search, Trophy, Sparkles, X, Check, Globe } from 'lucide-react';
import { CyberBadge } from '../atoms/CyberBadge';

interface PlayerAutocompleteProps {
  role: Role;
  currentSummoner?: string;
  onSelectPlayer: (player: PlayerSearchResult) => void;
  onClearPlayer?: () => void;
  onClose: () => void;
}

export const PlayerAutocomplete: React.FC<PlayerAutocompleteProps> = ({
  role,
  currentSummoner,
  onSelectPlayer,
  onClearPlayer,
  onClose,
}) => {
  const [query, setQuery] = useState(currentSummoner || '');
  const [suggestions, setSuggestions] = useState<PlayerSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load initial suggestions based on role
  useEffect(() => {
    setSuggestions(getRoleProSuggestions(role));
  }, [role]);

  // Debounced search
  useEffect(() => {
    let isCurrent = true;
    const timer = setTimeout(async () => {
      setIsLoading(true);
      try {
        const results = await searchPlayers(query, role);
        if (isCurrent) {
          setSuggestions(results);
          setSelectedIndex(0);
        }
      } finally {
        if (isCurrent) setIsLoading(false);
      }
    }, 150);

    return () => {
      isCurrent = false;
      clearTimeout(timer);
    };
  }, [query, role]);

  // Handle outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, suggestions.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + suggestions.length) % Math.max(1, suggestions.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (suggestions[selectedIndex]) {
        onSelectPlayer(suggestions[selectedIndex]);
        onClose();
      } else if (query.trim()) {
        onSelectPlayer({
          summonerName: query.trim(),
          tier: 'CHALLENGER',
          lp: 850,
          winrate: 57.5,
          totalGames: 210,
          kda: '3.6',
          isPro: false,
          mainRole: role,
        });
        onClose();
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div
      ref={containerRef}
      className="absolute top-0 left-12 z-50 w-72 bg-[#080d1a] border border-cyan-500/80 rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.8),0_0_20px_rgba(0,240,255,0.25)] backdrop-blur-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
    >
      {/* Search Input Header */}
      <div className="p-2 border-b border-slate-800 bg-[#050811] flex items-center gap-2">
        <Search className={`w-3.5 h-3.5 ${isLoading ? 'text-cyan-400 animate-spin' : 'text-slate-400'}`} />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`Chercher joueur ${role} (ex: Faker, Caps)...`}
          autoFocus
          className="flex-1 bg-transparent border-none text-xs font-mono text-cyan-300 placeholder-slate-500 focus:outline-none"
        />
        {query && (
          <button
            onClick={() => setQuery('')}
            className="text-slate-500 hover:text-slate-300 p-0.5"
          >
            <X className="w-3 h-3" />
          </button>
        )}
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-rose-400 p-0.5 ml-1"
          title="Fermer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Suggested Players List */}
      <div className="max-h-60 overflow-y-auto py-1 divide-y divide-slate-800/40">
        <div className="px-2.5 py-1 text-[10px] font-mono font-semibold tracking-wider text-slate-400 uppercase flex items-center justify-between bg-slate-950/40">
          <span>{query ? 'Résultats & Suggestions' : `Top Joueurs Pro (${role})`}</span>
          <span className="text-[9px] text-cyan-500">↑↓ Navigation • ↵ Valider</span>
        </div>

        {suggestions.length > 0 ? (
          suggestions.map((player, idx) => {
            const isSelected = idx === selectedIndex;
            return (
              <div
                key={`${player.summonerName}-${idx}`}
                onMouseDown={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onSelectPlayer(player);
                  onClose();
                }}
                onMouseEnter={() => setSelectedIndex(idx)}
                className={`flex items-center justify-between px-2.5 py-2 cursor-pointer transition-colors ${
                  isSelected ? 'bg-cyan-950/40 border-l-2 border-cyan-400' : 'hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <div className="w-6 h-6 rounded bg-slate-900 border border-slate-700 flex items-center justify-center shrink-0">
                    <User className="w-3.5 h-3.5 text-cyan-400" />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-mono font-bold text-slate-100 truncate">
                        {player.summonerName}
                      </span>
                      {player.proTeam && (
                        <span className="text-[9px] px-1 bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 rounded font-mono font-bold">
                          {player.proTeam}
                        </span>
                      )}
                      {player.country && (
                        <span className="text-[9px] font-mono text-slate-500">
                          {player.country}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                      <span className="text-amber-400 font-semibold">{player.tier}</span>
                      <span>{player.lp} LP</span>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col items-end shrink-0 pl-2 font-mono">
                  <span className="text-[11px] font-bold text-emerald-400">
                    {player.winrate.toFixed(1)}% WR
                  </span>
                  <span className="text-[9px] text-slate-500">
                    KDA {player.kda}
                  </span>
                </div>
              </div>
            );
          })
        ) : (
          <div className="p-3 text-center text-xs font-mono text-slate-500">
            Aucun profil exact trouvé. Appuyez sur Entrée pour utiliser "{query}".
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-1.5 border-t border-slate-800 bg-[#050811] flex items-center justify-between text-[10px] font-mono text-slate-400">
        {onClearPlayer && currentSummoner && (
          <button
            onClick={() => {
              onClearPlayer();
              onClose();
            }}
            className="text-rose-400 hover:text-rose-300 px-2 py-0.5 rounded hover:bg-rose-950/30 transition-colors"
          >
            Délier le joueur
          </button>
        )}
        <span className="ml-auto text-slate-500">ESC pour fermer</span>
      </div>
    </div>
  );
};
