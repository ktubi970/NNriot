import React, { useState } from 'react';
import { PlayerSlot, PlayerSearchResult } from '../../types';
import { RoleIcon } from '../atoms/RoleIcon';
import { Plus, User, X, History } from 'lucide-react';
import { PlayerAutocomplete } from './PlayerAutocomplete';

interface ChampionSlotProps {
  slot: PlayerSlot;
  team: 'blue' | 'red';
  onOpenChampionPicker: () => void;
  onUpdatePlayer: (slotId: string, updates: Partial<PlayerSlot>) => void;
  onClearChampion: () => void;
  onOpenHistory?: () => void;
}

export const ChampionSlot: React.FC<ChampionSlotProps> = ({
  slot,
  team,
  onOpenChampionPicker,
  onUpdatePlayer,
  onClearChampion,
  onOpenHistory,
}) => {
  const [isEditingPlayer, setIsEditingPlayer] = useState(false);
  const teamLabel = team === 'blue' ? 'équipe bleue' : 'équipe rouge';
  const playerName = slot.summonerName || slot.playerStats?.summonerName;
  const hasHistoryIdentity = Boolean(slot.playerStats?.tagLine || playerName?.includes('#'));

  const handleSelectPlayer = (player: PlayerSearchResult) => {
    onUpdatePlayer(slot.id, {
      summonerName: player.summonerName,
      playerStats: {
        summonerName: player.summonerName,
        tagLine: player.tagLine,
        puuid: player.puuid,
        source: player.source,
        tier: player.tier,
        rank: player.rank,
        lp: player.lp,
        winrate: player.winrate,
        totalGames: player.totalGames,
        kda: player.kda,
        isPro: player.isPro,
        proTeam: player.proTeam,
      },
    });
    setIsEditingPlayer(false);
  };

  const handleClearPlayer = () => {
    onUpdatePlayer(slot.id, { summonerName: '', playerStats: null });
    setIsEditingPlayer(false);
  };

  return (
    <div className="champion-slot">
      <div className="champion-role">
        <span aria-hidden="true"><RoleIcon role={slot.role} className="w-4 h-4" /></span>
        <span>{slot.role}</span>
      </div>

      <div className="min-w-0 flex-1">
        <button
          type="button"
          onClick={onOpenChampionPicker}
          className="champion-select w-full"
          aria-label={slot.champion
            ? `Changer ${slot.champion.name} · ${slot.role} · ${teamLabel}`
            : `Choisir un champion · ${slot.role} · ${teamLabel}`}
        >
          <span className="champion-avatar" aria-hidden="true">
            {slot.champion ? (
              <img
                src={slot.champion.iconUrl}
                alt=""
                onError={(event) => { event.currentTarget.style.visibility = 'hidden'; }}
              />
            ) : <Plus className="w-5 h-5" />}
          </span>
          <span className="min-w-0">
            <span className={`block text-sm font-medium truncate ${slot.champion ? '' : 'secondary-text'}`}>
              {slot.champion?.name || 'Choisir un champion'}
            </span>
            {slot.champion?.winRate != null && (
              <span className="block secondary-text text-xs">Taux de victoire : {slot.champion.winRate.toLocaleString('fr-FR', { maximumFractionDigits: 1 })} %</span>
            )}
          </span>
        </button>

        <div className="relative mt-1">
          <button
            type="button"
            onClick={() => setIsEditingPlayer(true)}
            className="flex items-center gap-1.5 min-h-9 w-full rounded-md text-left secondary-text hover:text-white text-xs pl-[3.75rem]"
            aria-label={`${playerName ? 'Changer le joueur' : 'Associer un joueur'} · ${slot.role} · ${teamLabel}`}
          >
            <User className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
            <span className="truncate">{playerName || 'Associer un joueur'}</span>
            {slot.playerStats?.proTeam && <span className="text-[10px] border border-[#464c57] rounded px-1 shrink-0">{slot.playerStats.proTeam}</span>}
          </button>
          {isEditingPlayer && (
            <PlayerAutocomplete
              role={slot.role}
              currentSummoner={slot.summonerName}
              onSelectPlayer={handleSelectPlayer}
              onClearPlayer={handleClearPlayer}
              onClose={() => setIsEditingPlayer(false)}
            />
          )}
        </div>
        {slot.playerStats?.winrate != null && (
          <p className="text-[11px] secondary-text pl-[3.75rem]">
            {slot.playerStats.winrate.toLocaleString('fr-FR', { maximumFractionDigits: 1 })} % de victoires
            {slot.playerStats.kda != null && ` · KDA ${slot.playerStats.kda}`}
          </p>
        )}
      </div>

      <div className="flex flex-col shrink-0">
        {slot.champion && (
          <button
            type="button"
            onClick={onClearChampion}
            className="icon-button"
            aria-label={`Retirer ${slot.champion.name} · ${slot.role} · ${teamLabel}`}
            title="Retirer le champion"
          >
            <X className="w-4 h-4" aria-hidden="true" />
          </button>
        )}
        {onOpenHistory && playerName && hasHistoryIdentity && (
          <button
            type="button"
            onClick={onOpenHistory}
            className="icon-button"
            aria-label={`Voir l’historique de ${playerName}`}
          >
            <History className="w-4 h-4" aria-hidden="true" />
          </button>
        )}
      </div>
    </div>
  );
};
