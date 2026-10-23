export type Role = 'TOP' | 'JUNGLE' | 'MID' | 'ADC' | 'SUPPORT';

export interface Champion {
  id: string;
  name: string;
  key: string;
  roles: Role[];
  iconUrl: string;
  winRate?: number;
  pickRate?: number;
  tags?: string[];
}

export interface PlayerStats {
  summonerName: string;
  tier?: string;
  rank?: string;
  lp?: number;
  winrate: number;
  totalGames: number;
  kda: string;
  championGames?: number;
  championWinrate?: number;
  isPro?: boolean;
  proTeam?: string;
}

export interface PlayerSearchResult {
  summonerName: string;
  tagLine?: string;
  tier: string;
  rank?: string;
  lp: number;
  winrate: number;
  totalGames: number;
  kda: string;
  isPro?: boolean;
  proTeam?: string;
  mainRole?: Role;
  country?: string;
}

export interface PlayerSlot {
  id: string;
  role: Role;
  champion: Champion | null;
  summonerName: string;
  playerStats: PlayerStats | null;
}

export interface FeatureImpact {
  id: string;
  name: string;
  impact: number; // positive = favors blue, negative = favors red
  category: 'matchup' | 'synergy' | 'scaling' | 'objective' | 'player';
  description: string;
  team: 'blue' | 'red';
}

export interface PredictionResult {
  blueWinProb: number; // e.g. 0.584 (58.4%)
  redWinProb: number;  // e.g. 0.416 (41.6%)
  confidence: number;  // e.g. 0.85
  blueExpectedKills?: number;
  redExpectedKills?: number;
  expectedDuration?: string;
  features: FeatureImpact[];
  latencyMs: number;
  modelType: 'neural_tf' | 'standalone_hashing';
  timestamp: string;
}

export interface SystemStatus {
  model_loaded: boolean;
  tensorflow_version: string;
  input_dimension: number | string;
  output_classes: number;
  status: string;
  latencyMs?: number;
  apiConnected?: boolean;
}

export interface AnalyticsChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  evidence?: {
    sampleSize?: number;
    confidenceInterval?: [number, number];
    qValue?: number;
    evidenceId?: string;
    patchAdjusted?: boolean;
  };
}

export interface PlayerMatchRecord {
  match_id: string;
  champion: string;
  role: Role;
  win: boolean;
  kills: number;
  deaths: number;
  assists: number;
  kda: string;
  cs: number;
  cs_per_min: string;
  gold: number;
  duration: string;
  game_mode: string;
  time_ago: string;
  items: number[];
}

export interface DatabaseStats {
  total_matches: number;
  total_training: number;
  untrained_records: number;
  total_players: number;
  crawl_queue_pending: number;
  db_size_mb: number;
  riot_api_configured: boolean;
}

export interface LiveGameInfo {
  gameId: string;
  gameMode: string;
  gameLength: number;
  blueTeam: { champion: string; summoner: string; role: Role }[];
  redTeam: { champion: string; summoner: string; role: Role }[];
}
