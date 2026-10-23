import React, { useState } from 'react';
import { PlayerSlot, Champion } from '../../types';
import { ChampionSlot } from '../molecules/ChampionSlot';
import { ChampionPickerModal } from './ChampionPickerModal';
import { MatchHistoryModal } from './MatchHistoryModal';
import { CyberBadge } from '../atoms/CyberBadge';
import { Shuffle, RotateCcw } from 'lucide-react';

interface TeamLineupProps {
  team: 'blue' | 'red';
  title: string;
  slots: PlayerSlot[];
  champions: Champion[];
  onUpdateSlot: (slotId: string, updates: Partial<PlayerSlot>) => void;
  onRandomizeTeam: () => void;
  onResetTeam: () => void;
}

export const TeamLineup: React.FC<TeamLineupProps> = ({
  team,
  title,
  slots,
  champions,
  onUpdateSlot,
  onRandomizeTeam,
  onResetTeam,
}) => {
  const isBlue = team === 'blue';
  const [activePickerSlot, setActivePickerSlot] = useState<PlayerSlot | null>(null);
  const [historyTarget, setHistoryTarget] = useState<{ name: string; tag: string } | null>(null);

  const assignedChampions = slots.filter(s => s.champion);
  const avgChampionWr = assignedChampions.length > 0
    ? (assignedChampions.reduce((acc, s) => acc + (s.champion?.winRate || 50), 0) / assignedChampions.length).toFixed(1)
    : '--';

  const handleOpenHistory = (slot: PlayerSlot) => {
    const name = slot.summonerName || slot.playerStats?.summonerName;
    if (name) {
      const tag = slot.playerStats?.proTeam || 'EUW';
      setHistoryTarget({ name, tag });
    }
  };

  return (
    <div className={`flex flex-col h-full bg-[#080d1a]/80 border ${isBlue ? 'border-cyan-500/30 shadow-[0_0_25px_rgba(0,240,255,0.08)]' : 'border-rose-500/30 shadow-[0_0_25px_rgba(244,63,94,0.08)]'} rounded-xl p-4 backdrop-blur-md`}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className={`w-3 h-3 rounded-sm rotate-45 ${isBlue ? 'bg-cyan-400 shadow-[0_0_8px_rgba(0,240,255,0.8)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]'}`} />
          <h3 className={`font-mono font-bold tracking-wider uppercase text-sm ${isBlue ? 'text-cyan-400' : 'text-rose-400'}`}>
            {title}
          </h3>
          <CyberBadge variant={isBlue ? 'cyan' : 'pink'} size="sm">
            {assignedChampions.length}/5 PICKS
          </CyberBadge>
        </div>

        <div className="flex items-center gap-2">
          <div className="text-right mr-1">
            <span className="text-[10px] font-mono text-slate-500 block">AVG WR</span>
            <span className={`text-xs font-mono font-semibold ${isBlue ? 'text-cyan-300' : 'text-rose-300'}`}>
              {avgChampionWr}%
            </span>
          </div>

          <button
            onClick={onRandomizeTeam}
            title="Randomize lineup"
            className="p-1.5 rounded border border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <Shuffle className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={onResetTeam}
            title="Reset lineup"
            className="p-1.5 rounded border border-slate-700 bg-slate-900/60 hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 5 Champion Slots */}
      <div className="flex flex-col gap-2 flex-1 justify-between">
        {slots.map((slot) => (
          <ChampionSlot
            key={slot.id}
            slot={slot}
            team={team}
            onOpenChampionPicker={() => setActivePickerSlot(slot)}
            onUpdatePlayer={onUpdateSlot}
            onClearChampion={() => onUpdateSlot(slot.id, { champion: null })}
            onOpenHistory={() => handleOpenHistory(slot)}
          />
        ))}
      </div>

      {/* Champion Picker Modal */}
      {activePickerSlot && (
        <ChampionPickerModal
          isOpen={Boolean(activePickerSlot)}
          onClose={() => setActivePickerSlot(null)}
          onSelect={(champion) => {
            onUpdateSlot(activePickerSlot.id, { champion });
            setActivePickerSlot(null);
          }}
          targetRole={activePickerSlot.role}
          champions={champions}
          currentChampionId={activePickerSlot.champion?.id}
        />
      )}

      {/* Match History Modal */}
      <MatchHistoryModal
        isOpen={Boolean(historyTarget)}
        summonerName={historyTarget?.name || ''}
        tagLine={historyTarget?.tag || 'EUW'}
        onClose={() => setHistoryTarget(null)}
      />
    </div>
  );
};
