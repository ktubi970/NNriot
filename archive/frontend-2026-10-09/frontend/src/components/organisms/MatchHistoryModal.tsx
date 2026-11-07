import React, { useState, useEffect } from 'react';
import { PlayerMatchRecord } from '../../types';
import { fetchPlayerHistory } from '../../services/api';
import { RoleIcon } from '../atoms/RoleIcon';
import { CyberBadge } from '../atoms/CyberBadge';
import { GlowButton } from '../atoms/GlowButton';
import { X, RefreshCw, Trophy, Swords, Clock, Coins, ShieldCheck, Flame, ExternalLink } from 'lucide-react';

interface MatchHistoryModalProps {
  isOpen: boolean;
  summonerName: string;
  tagLine?: string;
  onClose: () => void;
}

export const MatchHistoryModal: React.FC<MatchHistoryModalProps> = ({
  isOpen,
  summonerName,
  tagLine = 'EUW',
  onClose,
}) => {
  const [matches, setMatches] = useState<PlayerMatchRecord[]>([]);
  const [playerData, setPlayerData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncingRiot, setIsSyncingRiot] = useState(false);

  const loadHistory = async (syncRiot: boolean = false) => {
    if (!summonerName) return;
    if (syncRiot) setIsSyncingRiot(true);
    else setIsLoading(true);

    try {
      const data = await fetchPlayerHistory(summonerName, tagLine, 10, syncRiot);
      if (data) {
        setPlayerData(data.player);
        setMatches(data.matches || []);
      }
    } finally {
      setIsLoading(false);
      setIsSyncingRiot(false);
    }
  };

  useEffect(() => {
    if (isOpen && summonerName) {
      loadHistory(false);
    }
  }, [isOpen, summonerName, tagLine]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const winsCount = matches.filter(m => m.win).length;
  const winRate = matches.length > 0 ? Math.round((winsCount / matches.length) * 100) : 50;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/80 backdrop-blur-md transition-opacity"
        onClick={onClose}
      />

      {/* Modal Window */}
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-[#070b14] border border-cyan-500/50 rounded-2xl shadow-[0_0_50px_rgba(0,240,255,0.2)] flex flex-col overflow-hidden z-10 animate-in fade-in zoom-in-95 duration-200">
        {/* Header Strip */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#050811]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/40 flex items-center justify-center">
              <Trophy className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-black font-mono text-slate-100 tracking-wide">
                  {summonerName}
                </h2>
                <span className="text-xs font-mono text-slate-400">#{tagLine}</span>
                {playerData?.is_pro && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 rounded font-bold">
                    PRO PLAYER
                  </span>
                )}
              </div>
              <p className="text-xs font-mono text-slate-400">
                Historique des Matchs Récents & Performance SoloQ / Match-V5
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => loadHistory(true)}
              disabled={isSyncingRiot || isLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-cyan-950/40 hover:bg-cyan-900/60 border border-cyan-500/40 hover:border-cyan-400 text-cyan-300 rounded-lg text-xs font-mono transition-all disabled:opacity-50"
              title="Télécharger les derniers matchs depuis Riot API"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSyncingRiot ? 'animate-spin' : ''}`} />
              {isSyncingRiot ? 'SYNCHRONISATION...' : 'SYNCHRO RIOT API'}
            </button>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Quick Stats Overview Banner */}
        <div className="grid grid-cols-4 gap-3 p-4 bg-[#080e1d] border-b border-slate-800/80 font-mono text-xs">
          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex flex-col">
            <span className="text-[10px] text-slate-400 uppercase">TIER & RANG</span>
            <span className="text-amber-400 font-bold mt-0.5">
              {playerData?.tier || 'CHALLENGER'} {playerData?.lp ? `${playerData.lp} LP` : ''}
            </span>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex flex-col">
            <span className="text-[10px] text-slate-400 uppercase">WINRATE (10 MATCHS)</span>
            <span className={`font-bold mt-0.5 ${winRate >= 50 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {winRate}% ({winsCount}V - {matches.length - winsCount}D)
            </span>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex flex-col">
            <span className="text-[10px] text-slate-400 uppercase">KDA MOYEN</span>
            <span className="text-cyan-400 font-bold mt-0.5">
              {playerData?.avg_kda || '4.1'}
            </span>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 flex flex-col">
            <span className="text-[10px] text-slate-400 uppercase">TOTAL PARTIES</span>
            <span className="text-slate-200 font-bold mt-0.5">
              {playerData?.total_games || 340} Games
            </span>
          </div>
        </div>

        {/* Matches List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-16 text-slate-500 font-mono gap-3">
              <RefreshCw className="w-8 h-8 animate-spin text-cyan-400" />
              <span>Chargement des données Match-V5...</span>
            </div>
          ) : matches.length === 0 ? (
            <div className="text-center py-12 text-slate-500 font-mono text-xs">
              Aucun match enregistré pour ce profil. Cliquez sur "SYNCHRO RIOT API" pour ingérer les matchs.
            </div>
          ) : (
            matches.map((match) => (
              <div
                key={match.match_id}
                className={`flex items-center justify-between p-3 rounded-xl border transition-all ${
                  match.win
                    ? 'bg-cyan-950/15 border-cyan-500/30 hover:border-cyan-500/60 shadow-[0_0_15px_rgba(0,240,255,0.05)]'
                    : 'bg-rose-950/15 border-rose-500/30 hover:border-rose-500/60 shadow-[0_0_15px_rgba(244,63,94,0.05)]'
                }`}
              >
                {/* Result Tag & Champion */}
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`w-2 h-12 rounded-full ${
                      match.win ? 'bg-cyan-400 shadow-[0_0_8px_rgba(0,240,255,0.8)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]'
                    }`}
                  />

                  {/* Champion Icon */}
                  <div className="relative w-11 h-11 rounded-lg overflow-hidden border border-slate-700 shrink-0">
                    <img
                      src={`https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/${match.champion}.png`}
                      alt={match.champion}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/Aatrox.png';
                      }}
                    />
                  </div>

                  <div className="flex flex-col min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-mono font-bold ${match.win ? 'text-cyan-300' : 'text-rose-400'}`}>
                        {match.win ? 'VICTOIRE' : 'DÉFAITE'}
                      </span>
                      <span className="text-xs font-bold text-slate-100">{match.champion}</span>
                      <span className="text-[10px] font-mono text-slate-500">{match.game_mode}</span>
                    </div>

                    <div className="flex items-center gap-2 mt-0.5 text-[11px] font-mono text-slate-400">
                      <Clock className="w-3 h-3 text-slate-500" />
                      <span>{match.duration}</span>
                      <span className="text-slate-600">•</span>
                      <span>{match.time_ago}</span>
                    </div>
                  </div>
                </div>

                {/* Performance KDA & CS */}
                <div className="flex items-center gap-6 font-mono">
                  <div className="flex flex-col items-center">
                    <span className="text-xs font-bold text-slate-100">
                      {match.kills} / <span className="text-rose-400">{match.deaths}</span> / {match.assists}
                    </span>
                    <span className="text-[10px] text-cyan-400 font-semibold mt-0.5">
                      {match.kda} KDA
                    </span>
                  </div>

                  <div className="hidden sm:flex flex-col items-end text-right">
                    <span className="text-xs text-slate-300 font-medium">
                      {match.cs} CS ({match.cs_per_min}/m)
                    </span>
                    <span className="text-[10px] text-amber-400 flex items-center gap-1 mt-0.5">
                      <Coins className="w-3 h-3" />
                      {match.gold.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer info */}
        <div className="p-3 border-t border-slate-800 bg-[#050811] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Source: Base locale NNriot & Riot Games Match-V5</span>
          <button
            onClick={onClose}
            className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded transition-colors"
          >
            Fermer
          </button>
        </div>
      </div>
    </div>
  );
};
