import React, { useState, useEffect, useRef, useId } from 'react';
import { Role, PlayerSearchResult } from '../../types';
import { searchPlayers } from '../../services/api';
import { Search, X, User } from 'lucide-react';
import { AccessibleDialog } from '../atoms/AccessibleDialog';

interface PlayerAutocompleteProps {
  role: Role;
  currentSummoner?: string;
  onSelectPlayer: (player: PlayerSearchResult) => void;
  onClearPlayer?: () => void;
  onClose: () => void;
}

export const PlayerAutocomplete: React.FC<PlayerAutocompleteProps> = ({
  role, currentSummoner, onSelectPlayer, onClearPlayer, onClose,
}) => {
  const [query, setQuery] = useState(currentSummoner || '');
  const [suggestions, setSuggestions] = useState<PlayerSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const headingId = useId();
  const inputId = useId();
  const listId = useId();
  const hintId = useId();

  useEffect(() => {
    let current = true;
    setSuggestions([]);
    setSelectedIndex(0);
    setError(null);
    if (query.trim().length < 2) {
      setIsLoading(false);
      return () => { current = false; };
    }
    setIsLoading(true);
    const timer = window.setTimeout(async () => {
      try {
        const results = await searchPlayers(query.trim(), role);
        if (current) setSuggestions(results);
      } catch (cause) {
        if (current) setError(cause instanceof Error ? cause.message : 'La recherche de joueurs est indisponible.');
      } finally {
        if (current) setIsLoading(false);
      }
    }, 180);
    return () => { current = false; window.clearTimeout(timer); };
  }, [query, role]);

  const selectPlayer = (player: PlayerSearchResult) => {
    onSelectPlayer(player);
    onClose();
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (suggestions.length > 0) setSelectedIndex(previous => (previous + (event.key === 'ArrowDown' ? 1 : -1) + suggestions.length) % suggestions.length);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const player = suggestions[selectedIndex];
      if (player) selectPlayer(player);
    }
  };

  return (
    <AccessibleDialog isOpen onClose={onClose} labelledBy={headingId} describedBy={hintId} initialFocusRef={inputRef} className="max-w-lg">
      <header className="flex items-start justify-between gap-3 border-b border-slate-700/70 px-5 py-4">
        <div>
          <h2 id={headingId} className="text-lg font-semibold tracking-tight">Choisir un joueur</h2>
          <p id={hintId} className="mt-1 text-sm text-slate-400">Rechercher un profil enregistré pour le rôle {role}.</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Fermer la recherche de joueurs" className="icon-button shrink-0"><X aria-hidden="true" className="h-5 w-5" /></button>
      </header>
      <div className="border-b border-slate-700/70 p-4">
        <label htmlFor={inputId} className="mb-2 block text-sm text-slate-300">Rechercher un joueur</label>
        <div className="relative">
          <Search aria-hidden="true" className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input ref={inputRef} id={inputId} type="text" role="combobox" aria-autocomplete="list" aria-expanded={suggestions.length > 0} aria-controls={listId} aria-activedescendant={suggestions[selectedIndex] ? `${listId}-${selectedIndex}` : undefined} value={query} onChange={event => setQuery(event.target.value)} onKeyDown={handleKeyDown} placeholder="Nom du joueur…" className="field-input w-full pl-10" />
        </div>
        <p className="mt-2 text-xs text-slate-400">↑ ↓ pour parcourir les résultats · Entrée pour choisir</p>
      </div>
      <div className="min-h-0 overflow-y-auto p-3">
        {isLoading && <p role="status" className="p-4 text-sm text-slate-400">Recherche en cours…</p>}
        {error && <p role="alert" className="rounded-lg border border-amber-700/50 bg-amber-950/20 p-3 text-sm text-amber-200">{error}</p>}
        {!isLoading && !error && suggestions.length === 0 && <p role="status" className="p-4 text-sm text-slate-400">{query.trim().length < 2 ? 'Saisissez au moins deux caractères.' : 'Aucun profil trouvé. Essayez un autre nom.'}</p>}
        <div id={listId} role="listbox" aria-label="Profils trouvés" className="space-y-1">
          {suggestions.map((player, index) => (
            <button key={`${player.summonerName}-${player.tagLine ?? ''}-${index}`} id={`${listId}-${index}`} type="button" role="option" aria-selected={selectedIndex === index} onClick={() => selectPlayer(player)} onMouseEnter={() => setSelectedIndex(index)} className={`flex min-h-16 w-full items-center justify-between gap-3 rounded-xl border p-3 text-left transition-colors ${selectedIndex === index ? 'border-slate-400 bg-slate-700/40' : 'border-transparent hover:bg-slate-800/60'}`}>
              <span className="flex min-w-0 items-center gap-2">
                <User aria-hidden="true" className="h-4 w-4 shrink-0 text-slate-400" />
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium">{player.summonerName}{player.tagLine ? `#${player.tagLine}` : ''}</span>
                  <span className="mt-1 block text-xs text-slate-400">{player.tier || 'Rang non renseigné'}{typeof player.lp === 'number' ? ` · ${player.lp} LP` : ''}{player.proTeam ? ` · ${player.proTeam}` : ''}</span>
                </span>
              </span>
              {typeof player.winrate === 'number' && Number.isFinite(player.winrate) && <span className="shrink-0 text-xs text-slate-300">{player.winrate.toFixed(1)} % de victoires</span>}
            </button>
          ))}
        </div>
      </div>
      <footer className="flex items-center justify-between gap-3 border-t border-slate-700/70 px-4 py-3">
        {onClearPlayer && currentSummoner && <button type="button" onClick={() => { onClearPlayer(); onClose(); }} className="tool-button min-h-11">Retirer le joueur</button>}
        <span className="ml-auto text-xs text-slate-400">Échap pour fermer</span>
      </footer>
    </AccessibleDialog>
  );
};
