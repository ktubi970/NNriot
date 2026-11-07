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
  const teamLabel = isBlue ? 'équipe bleue' : 'équipe rouge';
  const [activePickerSlot, setActivePickerSlot] = useState<PlayerSlot | null>(null);
  const [historyTarget, setHistoryTarget] = useState<{ name: string; tag: string } | null>(null);
  const assignedChampions = slots.filter((slot) => slot.champion);
  const headingId = `${team}-team-heading`;

  const handleOpenHistory = (slot: PlayerSlot) => {
    const fullName = slot.summonerName || slot.playerStats?.summonerName;
    if (!fullName) return;
    const [name, nameTag] = fullName.split('#');
    const tag = slot.playerStats?.tagLine || nameTag;
    if (name && tag) setHistoryTarget({ name, tag });
  };

  return (
    <section className="surface-panel team-panel" aria-labelledby={headingId}>
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3">
        <div className="flex items-center gap-2.5">
          <span className={`h-2.5 w-2.5 rounded-full ${isBlue ? 'bg-[#80b5ff]' : 'bg-[#f4a1a7]'}`} aria-hidden="true" />
          <h2 id={headingId} className={`team-heading ${isBlue ? 'team-blue' : 'team-red'}`}>{title}</h2>
          <CyberBadge variant="slate">{assignedChampions.length}/5</CyberBadge>
        </div>
        <div className="flex items-center">
          <button
            type="button"
            onClick={onRandomizeTeam}
            aria-label={`Composer aléatoirement l’${teamLabel}`}
            title="Composer aléatoirement"
            className="icon-button"
          >
            <Shuffle className="w-4 h-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={onResetTeam}
            aria-label={`Réinitialiser l’${teamLabel}`}
            title="Réinitialiser l’équipe"
            className="icon-button"
            disabled={slots.every((slot) => !slot.champion && !slot.summonerName && !slot.playerStats)}
          >
            <RotateCcw className="w-4 h-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div>
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

      <MatchHistoryModal
        isOpen={Boolean(historyTarget)}
        summonerName={historyTarget?.name || ''}
        tagLine={historyTarget?.tag || ''}
        onClose={() => setHistoryTarget(null)}
      />
    </section>
  );
};
