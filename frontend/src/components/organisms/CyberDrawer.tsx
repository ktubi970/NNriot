import React, { useId, useRef } from 'react';
import { AnalyticsChatTab } from './AnalyticsChatTab';
import { LiveSpectatorTab } from './LiveSpectatorTab';
import { ModelTrainerTab } from './ModelTrainerTab';
import { X, MessageSquare, Eye, Sliders, History } from 'lucide-react';
import { AccessibleDialog } from '../atoms/AccessibleDialog';

interface CyberDrawerProps {
  isOpen: boolean;
  activeTab: 'analytics' | 'spectator' | 'trainer' | 'history' | null;
  onClose: () => void;
  onSelectTab: (tab: 'analytics' | 'spectator' | 'trainer' | 'history') => void;
  onImportLineup?: (blueChampions: string[], redChampions: string[]) => void;
}

const tabs = [
  { id: 'analytics', label: 'Analyse', icon: MessageSquare },
  { id: 'spectator', label: 'Partie en direct', icon: Eye },
  { id: 'trainer', label: 'Modèle', icon: Sliders },
  { id: 'history', label: 'Historique', icon: History },
] as const;

export const CyberDrawer: React.FC<CyberDrawerProps> = ({
  isOpen, activeTab, onClose, onSelectTab, onImportLineup,
}) => {
  const headingId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  const selectedTabRef = useRef<HTMLButtonElement>(null);

  return (
    <AccessibleDialog isOpen={isOpen && activeTab !== null} onClose={onClose} labelledBy={headingId} initialFocusRef={closeRef} variant="drawer">
      <header className="border-b border-slate-700/70 px-4 py-4">
        <div className="mb-3 flex items-center justify-between gap-4">
          <h2 id={headingId} className="text-lg font-semibold tracking-tight">Outils de la draft</h2>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Fermer les outils" className="icon-button shrink-0">
            <X aria-hidden="true" className="h-5 w-5" />
          </button>
        </div>
        <div role="tablist" aria-label="Outils" className="flex flex-wrap gap-1.5" onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const current = tabs.findIndex(tab => tab.id === activeTab);
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
            : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
          onSelectTab(tabs[next].id);
          requestAnimationFrame(() => selectedTabRef.current?.focus());
        }}>
          {tabs.map(({ id, label, icon: Icon }) => (
            <button key={id} ref={activeTab === id ? selectedTabRef : undefined} id={`${headingId}-${id}-tab`} type="button" role="tab" aria-selected={activeTab === id} aria-controls={`${headingId}-${id}-panel`} tabIndex={activeTab === id ? 0 : -1} onClick={() => onSelectTab(id)} className={`tool-button min-h-11 gap-2 ${activeTab === id ? 'bg-slate-600/60 text-white' : 'text-slate-300'}`}>
              <Icon aria-hidden="true" className="h-4 w-4" />
              {label}
            </button>
          ))}
        </div>
      </header>
      <div id={`${headingId}-${activeTab}-panel`} role="tabpanel" aria-labelledby={`${headingId}-${activeTab}-tab`} className="min-h-0 flex-1 overflow-hidden">
        {activeTab === 'analytics' && <AnalyticsChatTab />}
        {activeTab === 'spectator' && <LiveSpectatorTab onImportLineup={onImportLineup} />}
        {activeTab === 'trainer' && <ModelTrainerTab />}
        {activeTab === 'history' && <p className="p-5 text-sm text-slate-400">Sélectionnez un joueur pour consulter son historique.</p>}
      </div>
    </AccessibleDialog>
  );
};
