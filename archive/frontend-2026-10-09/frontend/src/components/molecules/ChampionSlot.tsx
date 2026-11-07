import React, { useState } from 'react';
import { PlayerSlot, Champion, PlayerSearchResult } from '../../types';
import { RoleIcon } from '../atoms/RoleIcon';
import { CyberBadge } from '../atoms/CyberBadge';
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

  const isBlue = team === 'blue';
  const teamBorder = isBlue ? 'hover:border-cyan-500/50 group-hover:border-cyan-500/40' : 'hover:border-rose-500/50 group-hover:border-rose-500/40';
  const teamBg = isBlue ? 'bg-cyan-950/10' : 'bg-rose-950/10';
  const teamAccent = isBlue ? 'text-cyan-400' : 'text-rose-400';

  const handleSelectPlayer = (player: PlayerSearchResult) => {
    onUpdatePlayer(slot.id, {
      summonerName: player.summonerName,
      playerStats: {
        summonerName: player.summonerName,
        tier: player.tier || 'CHALLENGER',
        rank: player.rank || 'I',
        lp: player.lp || 850,
        winrate: player.winrate || 55.0,
        totalGames: player.totalGames || 150,
        kda: player.kda || '3.5',
        isPro: player.isPro,
        proTeam: player.proTeam,
      },
    });
  };

  const handleClearPlayer = () => {
    onUpdatePlayer(slot.id, {
      summonerName: '',
      playerStats: null,
    });
  };

  return (
    <div
      className={`group relative flex items-center justify-between p-2.5 rounded-lg border border-slate-800/90 ${teamBg} ${teamBorder} backdrop-blur-sm transition-all duration-200 hover:shadow-lg`}
    >
      {/* Role Indicator & Avatar */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="flex flex-col items-center justify-center w-8 shrink-0">
          <RoleIcon role={slot.role} className="w-4 h-4 mb-1" />
          <span className="text-[9px] font-mono text-slate-400 tracking-wider">
            {slot.role}
          </span>
        </div>

        {/* Champion Avatar or Add Button */}
        {slot.champion ? (
          <div className="relative group/avatar cursor-pointer" onClick={onOpenChampionPicker}>
            <div className={`w-12 h-12 rounded-lg overflow-hidden border-2 ${isBlue ? 'border-cyan-500/80 shadow-[0_0_12px_rgba(0,240,255,0.3)]' : 'border-rose-500/80 shadow-[0_0_12px_rgba(244,63,94,0.3)]'} transition-transform group-hover/avatar:scale-105`}>
              <img
                src={slot.champion.iconUrl}
                alt={slot.champion.name}
                className="w-full h-full object-cover"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/Aatrox.png';
                }}
              />
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClearChampion();
              }}
              className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-slate-900 border border-slate-700 rounded-full flex items-center justify-center text-slate-400 hover:text-white hover:bg-rose-900 transition-colors"
              title="Remove champion"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          </div>
        ) : (
          <button
            onClick={onOpenChampionPicker}
            className={`w-12 h-12 rounded-lg border-2 border-dashed border-slate-700 hover:border-cyan-400/80 bg-slate-900/50 hover:bg-cyan-950/20 flex flex-col items-center justify-center text-slate-400 hover:text-cyan-300 transition-all group/btn`}
          >
            <Plus className="w-5 h-5 group-hover/btn:scale-110 transition-transform" />
          </button>
        )}

        {/* Champion Name & Player Info */}
        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <span
              onClick={onOpenChampionPicker}
              className={`text-sm font-semibold truncate cursor-pointer hover:underline ${
                slot.champion ? 'text-slate-100' : 'text-slate-500 font-mono text-xs'
              }`}
            >
              {slot.champion ? slot.champion.name : 'Pick Champion...'}
            </span>

            {slot.champion?.winRate && (
              <CyberBadge variant={slot.champion.winRate >= 50 ? 'green' : 'pink'} size="sm">
                {slot.champion.winRate.toFixed(1)}%
              </CyberBadge>
            )}
          </div>

          {/* Summoner / Player tag with Autocomplete */}
          <div className="relative">
            {isEditingPlayer ? (
              <PlayerAutocomplete
                role={slot.role}
                currentSummoner={slot.summonerName}
                onSelectPlayer={handleSelectPlayer}
                onClearPlayer={handleClearPlayer}
                onClose={() => setIsEditingPlayer(false)}
              />
            ) : (
              <div
                onClick={() => setIsEditingPlayer(true)}
                className="flex items-center gap-1.5 mt-0.5 cursor-pointer text-xs text-slate-400 hover:text-slate-200 group/player"
                title="Cliquer pour changer ou associer un joueur"
              >
                <User className="w-3 h-3 text-slate-500 group-hover/player:text-cyan-400" />
                <span className="truncate font-mono text-[11px] hover:text-cyan-300">
                  {slot.summonerName || (slot.playerStats ? slot.playerStats.summonerName : 'Associer Joueur')}
                </span>
                {slot.playerStats?.proTeam && (
                  <span className="text-[9px] px-1 bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 rounded font-mono font-bold">
                    {slot.playerStats.proTeam}
                  </span>
                )}
                {slot.playerStats?.tier && (
                  <span className="text-[9px] px-1 bg-slate-800 text-amber-300 rounded font-mono">
                    {slot.playerStats.tier.slice(0, 4)}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Player Stats Chip + History Button */}
      <div className="hidden sm:flex flex-col items-end text-right pl-2 shrink-0 gap-0.5">
        {slot.playerStats && (
          <>
            <span className="text-[10px] font-mono text-emerald-400">
              {slot.playerStats.winrate.toFixed(1)}% WR
            </span>
            <span className="text-[9px] font-mono text-slate-500">
              KDA {slot.playerStats.kda}
            </span>
          </>
        )}
        {onOpenHistory && (slot.summonerName || slot.playerStats?.summonerName) && (
          <button
            onClick={(e) => { e.stopPropagation(); onOpenHistory(); }}
            title="Voir l'historique des matchs"
            className="mt-0.5 p-0.5 rounded text-slate-500 hover:text-cyan-400 hover:bg-cyan-950/30 transition-colors"
          >
            <History className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
};
