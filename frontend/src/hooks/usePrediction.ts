import { useState, useEffect } from 'react';
import type { PlayerSlot, PredictionResult, Champion, Role } from '../types';
import { predictMatch } from '../services/api';

const DEFAULT_ROLES: Role[] = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'];
function createInitialSlots(prefix: 'blue' | 'red'): PlayerSlot[] {
  return DEFAULT_ROLES.map((role) => ({
    id: prefix + '_' + role.toLowerCase(), role, champion: null, summonerName: '', playerStats: null,
  }));
}

export function usePrediction(champions: Champion[]) {
  const [blueSlots, setBlueSlots] = useState<PlayerSlot[]>(() => createInitialSlots('blue'));
  const [redSlots, setRedSlots] = useState<PlayerSlot[]>(() => createInitialSlots('red'));
  const [result, setResult] = useState<{ key: string; prediction: PredictionResult } | null>(null);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const complete = [...blueSlots, ...redSlots].every((slot) => slot.champion !== null);
  const key = JSON.stringify([...blueSlots, ...redSlots].map((slot) => [slot.role, slot.champion?.name, slot.playerStats?.puuid]));

  useEffect(() => {
    setFailure(null);
    if (!complete) {
      setResult(null);
      setIsLoading(false);
      return;
    }
    const controller = new AbortController();
    setIsLoading(true);
    predictMatch(blueSlots, redSlots, { signal: controller.signal }).then((prediction) => {
      if (!controller.signal.aborted) setResult({ key, prediction });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setFailure({ key, message: error instanceof Error ? error.message : 'Analyse indisponible.' });
    }).finally(() => {
      if (!controller.signal.aborted) setIsLoading(false);
    });
    return () => controller.abort();
    // The key contains every inference input; name and display metadata do not trigger requests.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, complete]);

  const updateSlot = (team: 'blue' | 'red', slotId: string, updates: Partial<PlayerSlot>) => {
    const setter = team === 'blue' ? setBlueSlots : setRedSlots;
    setter((slots) => slots.map((slot) => slot.id === slotId ? { ...slot, ...updates } : slot));
  };

  const randomizeTeam = (team: 'blue' | 'red') => {
    if (!champions.length) return;
    const available = [...champions];
    for (let i = available.length - 1; i > 0; i--) {
      const next = Math.floor(Math.random() * (i + 1));
      [available[i], available[next]] = [available[next], available[i]];
    }
    const setter = team === 'blue' ? setBlueSlots : setRedSlots;
    setter((slots) => slots.map((slot) => {
      const index = Math.max(0, available.findIndex((champion) => champion.roles.includes(slot.role)));
      const champion = available.splice(index, 1)[0] ?? null;
      return { ...slot, champion };
    }));
  };

  const resetTeam = (team: 'blue' | 'red') => {
    (team === 'blue' ? setBlueSlots : setRedSlots)(createInitialSlots(team));
  };

  const swapTeams = () => {
    setBlueSlots(redSlots.map((slot) => ({ ...slot, id: slot.id.replace('red', 'blue') })));
    setRedSlots(blueSlots.map((slot) => ({ ...slot, id: slot.id.replace('blue', 'red') })));
  };

  const importLineup = (blueNames: string[], redNames: string[]) => {
    const byName = new Map(champions.map((champion) => [champion.name.toLowerCase(), champion]));
    const imported = (team: 'blue' | 'red', names: string[]) => createInitialSlots(team).map((slot, index) => ({
      ...slot, champion: byName.get(names[index]?.toLowerCase()) ?? null,
    }));
    setBlueSlots(imported('blue', blueNames));
    setRedSlots(imported('red', redNames));
  };

  return {
    blueSlots, redSlots,
    prediction: complete && result?.key === key ? result.prediction : null,
    error: complete && failure?.key === key ? failure.message : null,
    isLoading: complete && isLoading,
    updateSlot, randomizeTeam, resetTeam, swapTeams, importLineup,
  };
}
