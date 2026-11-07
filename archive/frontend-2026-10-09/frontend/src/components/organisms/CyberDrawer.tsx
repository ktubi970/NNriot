import React from 'react';
import { AnalyticsChatTab } from './AnalyticsChatTab';
import { LiveSpectatorTab } from './LiveSpectatorTab';
import { ModelTrainerTab } from './ModelTrainerTab';
import { X, MessageSquareCode, Eye, Sliders } from 'lucide-react';
import { CyberBadge } from '../atoms/CyberBadge';

interface CyberDrawerProps {
  isOpen: boolean;
  activeTab: 'analytics' | 'spectator' | 'trainer' | 'history' | null;
  onClose: () => void;
  onSelectTab: (tab: 'analytics' | 'spectator' | 'trainer' | 'history') => void;
  onImportLineup?: (blueChampions: string[], redChampions: string[]) => void;
}

export const CyberDrawer: React.FC<CyberDrawerProps> = ({
  isOpen,
  activeTab,
  onClose,
  onSelectTab,
  onImportLineup,
}) => {
  if (!isOpen || !activeTab) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-xl bg-[#070b14] border-l border-cyan-500/40 shadow-[-10px_0_40px_rgba(0,240,255,0.15)] flex flex-col">
          {/* Top Bar with Navigation Tabs */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-[#050811]">
            <div className="flex items-center gap-1">
              <button
                onClick={() => onSelectTab('analytics')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
                  activeTab === 'analytics'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <MessageSquareCode className="w-3.5 h-3.5" />
                CHAT IA
              </button>

              <button
                onClick={() => onSelectTab('spectator')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
                  activeTab === 'spectator'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Eye className="w-3.5 h-3.5" />
                SPECTATOR
              </button>

              <button
                onClick={() => onSelectTab('trainer')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono transition-all ${
                  activeTab === 'trainer'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                TRAINER
              </button>
            </div>

            <button
              onClick={onClose}
              aria-label="Close drawer"
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Active Tab View */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'analytics' && <AnalyticsChatTab />}
            {activeTab === 'spectator' && <LiveSpectatorTab onImportLineup={onImportLineup} />}
            {activeTab === 'trainer' && <ModelTrainerTab />}
          </div>
        </div>
      </div>
    </div>
  );
};
