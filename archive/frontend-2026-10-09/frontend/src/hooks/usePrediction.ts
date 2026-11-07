import { useState, useEffect, useRef, useCallback } from 'react';
import { PlayerSlot, PredictionResult, Champion, Role } from '../types';
import { predictMatch } from '../services/api';

const DEFAULT_ROLES: Role[] = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'];

function createInitialSlots(prefix: 'blue' | 'red'): PlayerSlot[] {
  return DEFAULT_ROLES.map((role) => ({
    id: `${prefix}_${role.toLowerCase()}`,
    role,
    champion: null,
    summonerName: '',
    playerStats: null,
  }));
}

export function usePrediction(champions: Champion[]) {
  const [blueSlots, setBlueSlots] = useState<PlayerSlot[]>(() => createInitialSlots('blue'));
  const [redSlots, setRedSlots] = useState<PlayerSlot[]>(() => createInitialSlots('red'));
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const debounceTimerRef = useRef<number | null>(null);

  // Auto-fill default iconic champions on initial load for instant demo feel
  useEffect(() => {
    if (champions.length > 0 && blueSlots.every((s) => !s.champion)) {
      const champMap = new Map(champions.map((c) => [c.name, c]));

      setBlueSlots([
        { id: 'blue_top', role: 'TOP', champion: champMap.get('Aatrox') || champions[0], summonerName: 'KCorp Caliste', playerStats: { summonerName: 'KCorp Caliste', tier: 'CHALLENGER', winrate: 56.4, totalGames: 310, kda: '3.4' } },
        { id: 'blue_jungle', role: 'JUNGLE', champion: champMap.get('Lee Sin') || champions[1], summonerName: 'Bo', playerStats: { summonerName: 'Bo', tier: 'CHALLENGER', winrate: 58.2, totalGames: 280, kda: '3.9' } },
        { id: 'blue_mid', role: 'MID', champion: champMap.get('Ahri') || champions[2], summonerName: 'Vetheo', playerStats: { summonerName: 'Vetheo', tier: 'CHALLENGER', winrate: 55.1, totalGames: 420, kda: '3.6' } },
        { id: 'blue_adc', role: 'ADC', champion: champMap.get('Jinx') || champions[3], summonerName: 'Upset', playerStats: { summonerName: 'Upset', tier: 'CHALLENGER', winrate: 57.0, totalGames: 390, kda: '4.2' } },
        { id: 'blue_support', role: 'SUPPORT', champion: champMap.get('Thresh') || champions[4], summonerName: 'Targamas', playerStats: { summonerName: 'Targamas', tier: 'CHALLENGER', winrate: 53.8, totalGames: 250, kda: '3.1' } },
      ]);

      setRedSlots([
        { id: 'red_top', role: 'TOP', champion: champMap.get('Renekton') || champions[5], summonerName: 'BrokenBlade', playerStats: { summonerName: 'BrokenBlade', tier: 'CHALLENGER', winrate: 55.5, totalGames: 340, kda: '3.3' } },
        { id: 'red_jungle', role: 'JUNGLE', champion: champMap.get('Sejuani') || champions[6], summonerName: 'Yike', playerStats: { summonerName: 'Yike', tier: 'CHALLENGER', winrate: 54.8, totalGames: 290, kda: '3.7' } },
        { id: 'red_mid', role: 'MID', champion: champMap.get('Sylas') || champions[7], summonerName: 'Caps', playerStats: { summonerName: 'Caps', tier: 'CHALLENGER', winrate: 59.2, totalGames: 510, kda: '4.1' } },
        { id: 'red_adc', role: 'ADC', champion: champMap.get('Ezreal') || champions[8], summonerName: 'Hans Sama', playerStats: { summonerName: 'Hans Sama', tier: 'CHALLENGER', winrate: 56.1, totalGames: 400, kda: '3.8' } },
        { id: 'red_support', role: 'SUPPORT', champion: champMap.get('Leona') || champions[9], summonerName: 'Mikyx', playerStats: { summonerName: 'Mikyx', tier: 'CHALLENGER', winrate: 55.0, totalGames: 330, kda: '3.0' } },
      ]);
    }
  }, [champions]);

  // Run live reactive prediction when slots change
  const triggerPrediction = useCallback(async (currentBlue: PlayerSlot[], currentRed: PlayerSlot[]) => {
    setIsLoading(true);
    try {
      const res = await predictMatch(currentBlue, currentRed);
      setPrediction(res);
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceTimerRef.current) {
      window.clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = window.setTimeout(() => {
      triggerPrediction(blueSlots, redSlots);
    }, 150);

    return () => {
      if (debounceTimerRef.current) {
        window.clearTimeout(debounceTimerRef.current);
      }
    };
  }, [blueSlots, redSlots, triggerPrediction]);

  // Helper actions
  const updateSlot = (team: 'blue' | 'red', slotId: string, updates: Partial<PlayerSlot>) => {
    if (team === 'blue') {
      setBlueSlots((prev) =>
        prev.map((s) => (s.id === slotId ? { ...s, ...updates } : s))
      );
    } else {
      setRedSlots((prev) =>
        prev.map((s) => (s.id === slotId ? { ...s, ...updates } : s))
      );
    }
  };

  const randomizeTeam = (team: 'blue' | 'red') => {
    if (!champions.length) return;
    const shuffled = [...champions].sort(() => Math.random() - 0.5);

    if (team === 'blue') {
      setBlueSlots((prev) =>
        prev.map((slot, i) => {
          const roleChamp = shuffled.find((c) => c.roles.includes(slot.role)) || shuffled[i];
          return { ...slot, champion: roleChamp };
        })
      );
    } else {
      setRedSlots((prev) =>
        prev.map((slot, i) => {
          const roleChamp = shuffled.find((c) => c.roles.includes(slot.role)) || shuffled[i + 5];
          return { ...slot, champion: roleChamp };
        })
      );
    }
  };

  const resetTeam = (team: 'blue' | 'red') => {
    if (team === 'blue') {
      setBlueSlots(createInitialSlots('blue'));
    } else {
      setRedSlots(createInitialSlots('red'));
    }
  };

  const swapTeams = () => {
    const tempBlue = blueSlots.map((s) => ({ ...s, id: s.id.replace('blue', 'red') }));
    const tempRed = redSlots.map((s) => ({ ...s, id: s.id.replace('red', 'blue') }));
    setBlueSlots(tempRed);
    setRedSlots(tempBlue);
  };

  const importLineup = (blueNames: string[], redNames: string[]) => {
    const champMap = new Map(champions.map((c) => [c.name.toLowerCase(), c]));

    setBlueSlots((prev) =>
      prev.map((slot, i) => {
        const name = blueNames[i]?.toLowerCase();
        const champ = champMap.get(name) || null;
        return { ...slot, champion: champ };
      })
    );

    setRedSlots((prev) =>
      prev.map((slot, i) => {
        const name = redNames[i]?.toLowerCase();
        const champ = champMap.get(name) || null;
        return { ...slot, champion: champ };
      })
    );
  };

  return {
    blueSlots,
    redSlots,
    prediction,
    isLoading,
    updateSlot,
    randomizeTeam,
    resetTeam,
    swapTeams,
    importLineup,
  };
}
