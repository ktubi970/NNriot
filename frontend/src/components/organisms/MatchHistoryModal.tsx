import React, { useState, useEffect, useCallback, useRef, useId } from 'react';
import { PlayerMatchRecord } from '../../types';
import { fetchPlayerHistory } from '../../services/api';
import { AccessibleDialog } from '../atoms/AccessibleDialog';
import { RoleIcon } from '../atoms/RoleIcon';
import { X, RefreshCw, Clock } from 'lucide-react';

interface MatchHistoryModalProps {
  isOpen: boolean;
  summonerName: string;
  tagLine?: string;
  onClose: () => void;
}

function displayValue(value: unknown): string {
  if (typeof value === 'number' && Number.isFinite(value)) return value.toLocaleString();
  if (typeof value === 'string' && value.trim()) return value;
  return 'Non renseigné';
}

export const MatchHistoryModal: React.FC<MatchHistoryModalProps> = ({
  isOpen, summonerName, tagLine = 'EUW', onClose,
}) => {
  const [matches, setMatches] = useState<PlayerMatchRecord[]>([]);
  const [playerData, setPlayerData] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncingRiot, setIsSyncingRiot] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pendingRef = useRef(false);
  const requestRef = useRef(0);
  const closeRef = useRef<HTMLButtonElement>(null);
  const headingId = useId();

  const loadHistory = useCallback(async (syncRiot = false) => {
    if (!summonerName || pendingRef.current) return;
    pendingRef.current = true;
    const request = ++requestRef.current;
    if (syncRiot) setIsSyncingRiot(true);
    else setIsLoading(true);
    setError(null);
    try {
      const data = await fetchPlayerHistory(summonerName, tagLine, 10, syncRiot);
      if (request !== requestRef.current) return;
      setPlayerData(data.player && typeof data.player === 'object' ? data.player : null);
      setMatches(Array.isArray(data.matches) ? data.matches : []);
    } catch (cause) {
      if (request !== requestRef.current) return;
      setError(cause instanceof Error ? cause.message : 'L’historique est indisponible. Réessayez plus tard.');
    } finally {
      if (request === requestRef.current) {
        pendingRef.current = false;
        setIsLoading(false);
        setIsSyncingRiot(false);
      }
    }
  }, [summonerName, tagLine]);

  useEffect(() => {
    requestRef.current += 1;
    pendingRef.current = false;
    setMatches([]);
    setPlayerData(null);
    setError(null);
    setIsLoading(false);
    setIsSyncingRiot(false);
    if (isOpen && summonerName) void loadHistory();
    return () => { requestRef.current += 1; pendingRef.current = false; };
  }, [isOpen, summonerName, loadHistory]);

  const wins = matches.filter(match => match.win).length;
  const winRate = matches.length > 0 ? Math.round(wins / matches.length * 100) : null;
  const summary = [
    { label: 'Rang', value: playerData?.tier ? `${displayValue(playerData.tier)}${typeof playerData.lp === 'number' ? ` · ${playerData.lp} LP` : ''}` : 'Non renseigné' },
    { label: 'Victoires récentes', value: winRate === null ? 'Aucun match' : `${winRate} % · ${wins}/${matches.length}` },
    { label: 'KDA moyen', value: displayValue(playerData?.avg_kda) },
    { label: 'Parties enregistrées', value: displayValue(playerData?.total_games) },
  ];

  return (
    <AccessibleDialog isOpen={isOpen} onClose={onClose} labelledBy={headingId} initialFocusRef={closeRef} className="max-w-3xl">
      <header className="flex items-start justify-between gap-3 border-b border-slate-700/70 px-5 py-4">
        <div className="min-w-0">
          <h2 id={headingId} className="text-lg font-semibold tracking-tight">Historique de {summonerName}</h2>
          <p className="mt-1 text-sm text-slate-400">{tagLine ? `#${tagLine} · ` : ''}Matchs récents</p>
        </div>
        <button ref={closeRef} type="button" onClick={onClose} aria-label="Fermer l’historique" className="icon-button shrink-0"><X aria-hidden="true" className="h-5 w-5" /></button>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-slate-400">Les données proviennent du service d’historique.</p>
          <button type="button" onClick={() => void loadHistory(true)} disabled={isLoading || isSyncingRiot || !summonerName} className="tool-button min-h-11 gap-2 disabled:opacity-50">
            <RefreshCw aria-hidden="true" className="h-4 w-4" />
            {isSyncingRiot ? 'Synchronisation…' : 'Synchroniser avec Riot'}
          </button>
        </div>
        {error && <div role="alert" className="rounded-xl border border-amber-700/50 bg-amber-950/20 p-4 text-sm text-amber-200"><p>{error}</p><button type="button" onClick={() => void loadHistory()} disabled={isLoading || isSyncingRiot} className="tool-button mt-3 min-h-11">Réessayer</button></div>}
        {isLoading ? <p role="status" className="py-10 text-center text-sm text-slate-400">Chargement de l’historique…</p> : (
          <>
            {matches.length > 0 || playerData ? <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">{summary.map(stat => <div key={stat.label} className="rounded-xl border border-slate-700/70 bg-slate-900/30 p-3"><dt className="text-xs text-slate-400">{stat.label}</dt><dd className="mt-2 text-sm font-medium">{stat.value}</dd></div>)}</dl> : null}
            {matches.length === 0 && !error ? <p role="status" className="py-10 text-center text-sm text-slate-400">Aucun match enregistré pour ce profil.</p> : null}
            <ul className="space-y-2" aria-label="Matchs récents">
              {matches.map(match => (
                <li key={match.match_id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-700/70 bg-slate-900/30 p-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                      <span className="rounded bg-slate-700/50 px-2 py-1 font-medium">{match.win ? 'Victoire' : 'Défaite'}</span>
                      <strong className="font-semibold">{match.champion}</strong>
                      <RoleIcon role={match.role} className="h-4 w-4" />
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400"><Clock aria-hidden="true" className="h-3.5 w-3.5" /><span>{displayValue(match.duration)}</span><span>{match.time_ago}</span></div>
                  </div>
                  <div className="text-right text-sm">
                    <p>{displayValue(match.kills)} / {displayValue(match.deaths)} / {displayValue(match.assists)}</p>
                    <p className="mt-1 text-xs text-slate-400">KDA {displayValue(match.kda)} · CS {displayValue(match.cs)}</p>
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
      <footer className="flex justify-end border-t border-slate-700/70 px-5 py-3"><button type="button" onClick={onClose} className="tool-button min-h-11">Fermer</button></footer>
    </AccessibleDialog>
  );
};
