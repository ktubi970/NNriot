import type {
  PlayerSlot, PredictionResult, SystemStatus, PlayerMatchRecord,
  DatabaseStats, Role, PlayerSearchResult, TeamProbabilities,
} from '../types';

export const API_BASE = '';
type JsonObject = Record<string, unknown>;
const object = (value: unknown): JsonObject =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {};
const number = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const string = (value: unknown): string | undefined => typeof value === 'string' ? value : undefined;

async function requestJson(url: string, options?: RequestInit): Promise<unknown> {
  const response = await fetch(API_BASE + url, options);
  const data: unknown = await response.json();
  const fields = object(data);
  if (!response.ok || fields.success === false) {
    const messages: Record<string, string> = {
      model_contract: 'Le modèle est indisponible pour cette draft.',
      feature_unavailable: 'Les données historiques nécessaires sont indisponibles.',
      feature_contract: 'Cette draft ne satisfait pas les conditions du modèle.',
    };
    const code = string(fields.error_code) ?? '';
    throw new Error(messages[code] ?? string(fields.error) ?? string(fields.answer) ?? 'Le service est indisponible. Réessayez plus tard.');
  }
  return data;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const start = performance.now();
  try {
    const data = object(await requestJson('/api/status'));
    const validation = object(data.serving_validation);
    return {
      model_loaded: data.model_loaded === true,
      tensorflow_version: string(data.tensorflow_version) ?? null,
      input_dimension: number(data.input_dimension) ?? string(data.input_dimension) ?? null,
      output_classes: number(data.output_classes) ?? null,
      status: string(data.status) ?? 'État du modèle inconnu',
      latencyMs: Math.round(performance.now() - start),
      apiConnected: true,
      model_id: string(data.model_id),
      feature_contract: string(data.feature_contract),
      calibration_available: data.calibration_available === true,
      serving_validation: string(validation.status) ? { status: String(validation.status) } : undefined,
      trainer_loaded: data.trainer_loaded === true,
    };
  } catch {
    return {
      model_loaded: false, tensorflow_version: null, input_dimension: null,
      output_classes: null, status: 'Service indisponible', apiConnected: false,
    };
  }
}

export function getRoleProSuggestions(_role: Role): PlayerSearchResult[] {
  return [];
}

export async function searchPlayers(query: string, role?: Role): Promise<PlayerSearchResult[]> {
  if (query.trim().length < 2) return [];
  const data = await requestJson('/api/players/search?q=' + encodeURIComponent(query.trim()));
  if (!Array.isArray(data)) throw new Error('La recherche a retourné un format inattendu.');
  return data.map(object).filter((player) => string(player.puuid) && (string(player.game_name) || string(player.summonerName)))
    .map((player) => ({
      summonerName: string(player.game_name) ?? String(player.summonerName),
      puuid: String(player.puuid), source: 'database' as const,
      tagLine: string(player.tag_line),
      tier: string(player.tier), rank: string(player.rank), lp: number(player.lp),
      winrate: number(player.winrate), totalGames: number(player.total_matches) ?? number(player.total_games) ?? number(player.totalGames),
      kda: string(player.avg_kda) ?? string(player.kda) ?? (number(player.avg_kda) !== undefined ? String(player.avg_kda) : number(player.kda) !== undefined ? String(player.kda) : undefined),
      isPro: player.is_pro === true, proTeam: string(player.team), mainRole: role,
    }));
}

function probabilities(value: unknown): TeamProbabilities {
  const pair = object(value);
  const blue = number(pair.team_a);
  const red = number(pair.team_b);
  if (blue === undefined || red === undefined || blue < 0 || blue > 1 || red < 0 || red > 1 || Math.abs(blue + red - 1) > 0.001) {
    throw new Error('Le service a retourné des probabilités invalides.');
  }
  return { blue, red };
}

const canonicalRoles: Record<Role, string> = {
  TOP: 'TOP', JUNGLE: 'JUNGLE', MID: 'MIDDLE', ADC: 'BOTTOM', SUPPORT: 'UTILITY',
};

export async function predictMatch(
  blueSlots: PlayerSlot[], redSlots: PlayerSlot[], options?: { signal?: AbortSignal },
): Promise<PredictionResult> {
  if (blueSlots.length !== 5 || redSlots.length !== 5 || [...blueSlots, ...redSlots].some((slot) => !slot.champion)) {
    throw new Error('Choisissez les dix champions avant de lancer une analyse.');
  }
  const start = performance.now();
  const payload = (slots: PlayerSlot[]) => slots.map((slot) => ({
    puuid: slot.playerStats?.puuid,
    champion_name: slot.champion!.name,
    role: canonicalRoles[slot.role],
  }));
  const data = object(await requestJson('/api/predict/custom', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: options?.signal,
    body: JSON.stringify({ blue_team: payload(blueSlots), red_team: payload(redSlots), game_mode: 'CLASSIC' }),
  }));
  const raw = probabilities(data.probabilities_raw);
  const calibrated = data.probabilities_calibrated == null ? null : probabilities(data.probabilities_calibrated);
  const calibrationStatus = string(object(data.calibration).status) ?? 'unavailable';
  const displayed = calibrated ?? raw;
  const decision = object(data.selective_decision);
  const conviction = number(decision.conviction);
  const threshold = number(decision.threshold_applied);
  const actionable = calibrated !== null && calibrationStatus === 'artifact_verified'
    && decision.is_actionable === true && decision.abstained === false
    && conviction !== undefined && conviction >= 0 && conviction <= 0.5
    && threshold !== undefined && threshold >= 0 && threshold <= 0.5 && conviction >= threshold;
  const recommendation = actionable
    ? 'La conviction atteint le seuil du service.'
    : calibrated === null || calibrationStatus !== 'artifact_verified'
      ? 'Calibration indisponible ou non vérifiée : aucune conclusion exploitable.'
      : decision.tier === 'UNVALIDATED_MODEL'
        ? 'La validation indépendante du modèle est indisponible.'
        : conviction !== undefined && threshold !== undefined && conviction < threshold
          ? 'La conviction est inférieure au seuil du service.'
          : 'Le service ne confirme pas de décision exploitable.';
  const priors = object(data.priors_provenance);
  return {
    blueWinProb: displayed.blue, redWinProb: displayed.red, confidence: Math.max(displayed.blue, displayed.red),
    rawProbabilities: raw, calibratedProbabilities: calibrated, calibrationStatus,
    selectiveDecision: {
      conviction: conviction !== undefined && conviction >= 0 && conviction <= 0.5 ? conviction : null,
      tier: string(decision.tier) ?? 'UNAVAILABLE', isActionable: actionable, abstained: !actionable,
      thresholdApplied: threshold !== undefined && threshold >= 0 && threshold <= 0.5 ? threshold : null,
      recommendation,
    },
    modelId: string(object(data.model).id) ?? null,
    draftAdvantage: data.draft_advantage == null ? null : probabilities(data.draft_advantage),
    features: [], explanationMethod: null,
    historyAvailable: priors.history_available === true, priorPolicy: string(priors.policy) ?? null,
    latencyMs: Math.round(performance.now() - start), modelType: 'neural_tf', timestamp: new Date().toLocaleTimeString('fr-FR'),
  };
}

export async function queryAnalyticsChat(question: string) {
  return object(await requestJson('/api/analytics/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: question }),
  }));
}

export async function triggerManualTraining() {
  return object(await requestJson('/api/train/manual', { method: 'POST' }));
}

export async function fetchPlayerHistory(
  name: string, tag = 'EUW', count = 10, syncRiot = false,
): Promise<{ player: Record<string, unknown>; matches: PlayerMatchRecord[] }> {
  const url = '/api/player/history?name=' + encodeURIComponent(name) + '&tag=' + encodeURIComponent(tag)
    + '&count=' + count + '&sync_riot=' + (syncRiot ? 1 : 0);
  const data = object(await requestJson(url));
  if (!Array.isArray(data.matches)) throw new Error('Historique indisponible.');
  const displayRoles: Record<string, Role> = { MIDDLE: 'MID', BOTTOM: 'ADC', UTILITY: 'SUPPORT' };
  const matches = data.matches.map((value) => {
    const match = object(value);
    const role = string(match.role) ?? 'UNKNOWN';
    return { ...match, role: displayRoles[role] ?? role };
  }) as unknown as PlayerMatchRecord[];
  return { player: object(data.player), matches };
}

export async function fetchDatabaseStats(): Promise<DatabaseStats> {
  return await requestJson('/api/db/stats') as DatabaseStats;
}

export async function triggerDatabaseCollect(gameName: string, tagLine: string, count = 10) {
  return object(await requestJson('/api/history/collect', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ game_name: gameName, tag_line: tagLine, count }),
  }));
}

export async function fetchCollectStatus() {
  return object(await requestJson('/api/history/status'));
}

export async function triggerBatchCollect(playersText: string, count = 20) {
  return object(await requestJson('/api/batch/collect', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ players: playersText, count, sources: ['lolpros'] }),
  }));
}

export async function fetchBatchStatus() {
  return object(await requestJson('/api/batch/status'));
}
