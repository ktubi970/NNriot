// Meta Research page data types

export interface ChampionWinrateByPatch {
  patch: string;
  winrate: number;
  games: number;
}

export interface MatchupStat {
  champion_a: string;
  champion_b: string;
  lane: string;
  winrate_a: number;
  games: number;
}

export interface ProTrend {
  champion: string;
  picks: number;
  bans: number;
  wins: number;
  presence: number; // (picks + bans) / total_games * 100
  winrate: number;
}

export interface ObjectiveCorrelation {
  objective: 'first_dragon' | 'first_baron' | 'first_tower';
  winrate_when_taken: number;
  games_sampled: number;
}

export interface TierListEntry {
  champion: string;
  lane: string;
  tier: 'S' | 'A' | 'B' | 'C';
  winrate: number;
  pickrate: number;
  games: number;
}

export interface DraftSosResult {
  counter_lineup: string[];
  historical_winrate_vs: number;
  games_sampled: number;
}

export type MetaLane = 'TOP' | 'JUNGLE' | 'MID' | 'ADC' | 'SUPPORT';
