import React, { useState, useMemo, useEffect, useRef, useId } from 'react';
import { Champion, Role } from '../../types';
import { Search, X } from 'lucide-react';
import { AccessibleDialog } from '../atoms/AccessibleDialog';
import { RoleIcon } from '../atoms/RoleIcon';

interface ChampionPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (champion: Champion) => void;
  targetRole: Role;
  champions: Champion[];
  currentChampionId?: string;
}

const roleLabels: Record<Role | 'ALL', string> = {
  ALL: 'Tous', TOP: 'Top', JUNGLE: 'Jungle', MID: 'Mid', ADC: 'ADC', SUPPORT: 'Support',
};

export const ChampionPickerModal: React.FC<ChampionPickerModalProps> = ({
  isOpen, onClose, onSelect, targetRole, champions, currentChampionId,
}) => {
  const [search, setSearch] = useState('');
  const [selectedRole, setSelectedRole] = useState<Role | 'ALL'>(targetRole);
  const inputRef = useRef<HTMLInputElement>(null);
  const headingId = useId();
  const searchId = useId();

  useEffect(() => {
    if (isOpen) {
      setSelectedRole(targetRole);
      setSearch('');
    }
  }, [isOpen, targetRole]);

  const filteredChampions = useMemo(() => champions.filter(champion =>
    champion.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())
    && (selectedRole === 'ALL' || champion.roles.includes(selectedRole)),
  ), [champions, search, selectedRole]);

  return (
    <AccessibleDialog isOpen={isOpen} onClose={onClose} labelledBy={headingId} initialFocusRef={inputRef} className="max-w-3xl">
      <header className="flex items-center justify-between gap-4 border-b border-slate-700/70 px-5 py-4">
        <div>
          <h2 id={headingId} className="text-lg font-semibold tracking-tight">Choisir un champion</h2>
          <p className="mt-1 text-sm text-slate-400">Emplacement {roleLabels[targetRole]}</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Fermer le sélecteur de champions" className="icon-button shrink-0">
          <X aria-hidden="true" className="h-5 w-5" />
        </button>
      </header>

      <div className="space-y-3 border-b border-slate-700/70 p-4">
        <label htmlFor={searchId} className="sr-only">Rechercher un champion</label>
        <div className="relative">
          <Search aria-hidden="true" className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input ref={inputRef} id={searchId} type="text" value={search} onChange={event => setSearch(event.target.value)} placeholder="Rechercher un champion" className="field-input w-full pl-10" />
        </div>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filtrer les champions par rôle">
          {(['ALL', 'TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'] as const).map(role => (
            <button key={role} type="button" onClick={() => setSelectedRole(role)} aria-pressed={selectedRole === role} className={`tool-button min-h-10 gap-1.5 ${selectedRole === role ? 'bg-slate-600/60 text-white' : 'text-slate-300'}`}>
              {role !== 'ALL' && <span aria-hidden="true"><RoleIcon role={role} className="h-3.5 w-3.5" /></span>}
              {roleLabels[role]}
            </button>
          ))}
        </div>
      </div>

      <div className="grid min-h-0 grid-cols-3 gap-2.5 overflow-y-auto p-4 sm:grid-cols-4 md:grid-cols-6" aria-label="Champions disponibles">
        {filteredChampions.map(champion => (
          <button key={champion.id} type="button" onClick={() => { onSelect(champion); onClose(); }} aria-pressed={champion.id === currentChampionId} className={`group flex min-h-28 flex-col items-center rounded-xl border p-2.5 text-sm transition-colors ${champion.id === currentChampionId ? 'border-slate-300 bg-slate-600/40' : 'border-slate-700/70 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-800'}`}>
            <img src={champion.iconUrl} alt="" className="h-14 w-14 rounded-lg border border-slate-700 object-cover" onError={event => { event.currentTarget.style.visibility = 'hidden'; }} />
            <span className="mt-2 max-w-full truncate font-medium">{champion.name}</span>
            {typeof champion.winRate === 'number' && Number.isFinite(champion.winRate) && <span className="mt-1 text-xs text-slate-400">{champion.winRate.toFixed(1)} % de victoires</span>}
          </button>
        ))}
        {filteredChampions.length === 0 && <p role="status" className="col-span-full py-10 text-center text-sm text-slate-400">{champions.length === 0 ? 'Le catalogue des champions est indisponible.' : 'Aucun champion ne correspond à ces filtres.'}</p>}
      </div>

      <footer className="flex items-center justify-between gap-3 border-t border-slate-700/70 px-5 py-3 text-xs text-slate-400">
        <span aria-live="polite">{filteredChampions.length} champion{filteredChampions.length > 1 ? 's' : ''}</span>
        <span>Échap pour fermer</span>
      </footer>
    </AccessibleDialog>
  );
};
