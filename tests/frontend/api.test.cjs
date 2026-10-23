const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

const source = fs.readFileSync(path.resolve(__dirname, '../../frontend/src/services/api.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;

function apiWith(fetch) {
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports, fetch, performance, console,
    Date, AbortController, setTimeout, clearTimeout,
  });
  return module.exports;
}

const roles = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'];
function lineup(team) {
  return roles.map((role, i) => ({
    id: `${team}_${role}`, role, champion: { name: `Champion${i}`, key: String(i + 1) },
    summonerName: 'Visible name',
    playerStats: i === 0 ? { summonerName: 'Visible name', puuid: 'actual-player-id' } : null,
  }));
}
function response(overrides = {}) {
  return {
    success: true,
    blue_win_probability: 0.62, red_win_probability: 0.38,
    confidence: 0.62,
    probabilities_raw: { team_a: 0.62, team_b: 0.38 },
    probabilities_calibrated: null,
    calibration: { status: 'unavailable', temperature: null },
    selective_decision: {
      conviction: 0.12, tier: 'UNCALIBRATED', is_actionable: false,
      abstained: true, threshold_applied: 0.1, recommendation: 'Abstain - winner calibration unavailable',
    },
    model: { id: 'legacy-14d', feature_contract: 'nnriot.players14.v1' },
    ...overrides,
  };
}
const ok = (data) => ({ ok: true, json: async () => data });

test('unavailable inference rejects instead of generating a convincing outcome', async () => {
  const api = apiWith(async () => ({ ok: false, status: 503, json: async () => ({ error: 'Unavailable', error_code: 'model_contract' }) }));
  await assert.rejects(api.predictMatch(lineup('blue'), lineup('red')));
});

test('uncalibrated service probabilities preserve abstention without fabricated metrics or explanations', async () => {
  const api = apiWith(async () => ok(response()));
  const result = await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(result.blueWinProb, 0.62);
  assert.equal(result.calibratedProbabilities, null);
  assert.equal(result.selectiveDecision.abstained, true);
  assert.equal(result.selectiveDecision.isActionable, false);
  assert.equal(result.selectiveDecision.tier, 'UNCALIBRATED');
  assert.equal(result.blueExpectedKills, undefined);
  assert.equal(result.expectedDuration, undefined);
  assert.equal(result.features.length, 0);
});

test('calibrated probabilities are displayed while raw probabilities remain available', async () => {
  const api = apiWith(async () => ok(response({
    probabilities_calibrated: { team_a: 0.55, team_b: 0.45 },
    calibration: { status: 'artifact_verified', temperature: 1.7 },
    selective_decision: { conviction: 0.05, tier: 'ABSTAIN', is_actionable: false, abstained: true, threshold_applied: 0.1 },
  })));
  const result = await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(result.blueWinProb, 0.55);
  assert.equal(result.redWinProb, 0.45);
  assert.equal(result.rawProbabilities.blue, 0.62);
  assert.equal(result.selectiveDecision.conviction, 0.05);
});

test('missing selective contract fails closed even when the service winner is confident', async () => {
  const api = apiWith(async () => ok(response({ selective_decision: undefined })));
  const result = await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(result.selectiveDecision.abstained, true);
  assert.equal(result.selectiveDecision.isActionable, false);
});

test('closed decision never repeats an actionable recommendation when calibration is missing', async () => {
  const api = apiWith(async () => ok(response({
    selective_decision: { conviction: 0.12, tier: 'HIGH_CONFIDENCE', is_actionable: true, abstained: false, threshold_applied: 0.1, recommendation: 'Bet on TEAM_A' },
  })));
  const result = await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(result.selectiveDecision.abstained, true);
  assert.doesNotMatch(result.selectiveDecision.recommendation, /bet on|parier/i);
});

test('malformed probability pairs are rejected instead of displayed', async () => {
  const api = apiWith(async () => ok(response({ probabilities_raw: { team_a: 0.7, team_b: 0.7 } })));
  await assert.rejects(api.predictMatch(lineup('blue'), lineup('red')));
});

test('payload preserves real identities, leaves unknown players anonymous, and canonizes roles', async () => {
  let payload;
  const api = apiWith(async (_url, options) => { payload = JSON.parse(options.body); return ok(response()); });
  await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(payload.blue_team[0].puuid, 'actual-player-id');
  assert.equal(payload.blue_team[1].puuid, undefined);
  assert.deepEqual(payload.blue_team.map((slot) => slot.role), ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']);
});

test('zero win probability is a valid service result', async () => {
  const api = apiWith(async () => ok(response({ probabilities_raw: { team_a: 0, team_b: 1 } })));
  const result = await api.predictMatch(lineup('blue'), lineup('red'));
  assert.equal(result.blueWinProb, 0);
  assert.equal(result.redWinProb, 1);
});

test('unreachable status leaves model dimensions and telemetry unknown', async () => {
  const api = apiWith(async () => { throw new TypeError('Failed to fetch'); });
  const result = await api.getSystemStatus();
  assert.equal(result.apiConnected, false);
  assert.equal(result.input_dimension, null);
  assert.equal(result.latencyMs, undefined);
});

test('failed player lookup never invents a profile', async () => {
  const api = apiWith(async () => ({ ok: false, status: 503, json: async () => ({ error: 'unavailable' }) }));
  await assert.rejects(api.searchPlayers('Unknown player', 'MID'));
});

test('database player lookup preserves supplied aggregate stats and zero values', async () => {
  const api = apiWith(async () => ok([{ puuid: 'real-id', game_name: 'Known', tag_line: 'EUW', total_matches: 24, avg_kda: 2.75, winrate: 0, lp: 0 }]));
  const result = await api.searchPlayers('Known', 'MID');
  assert.equal(result[0].puuid, 'real-id');
  assert.equal(result[0].totalGames, 24);
  assert.equal(result[0].kda, '2.75');
  assert.equal(result[0].winrate, 0);
  assert.equal(result[0].lp, 0);
  assert.equal(result[0].tier, undefined);
});

test('failed analytics, history and database requests never invent evidence or records', async () => {
  const api = apiWith(async () => ({ ok: false, status: 503, json: async () => ({ error: 'unavailable' }) }));
  await assert.rejects(api.queryAnalyticsChat('Dragon?'));
  await assert.rejects(api.fetchPlayerHistory('Someone'));
  await assert.rejects(api.fetchDatabaseStats());
});

test('history maps backend lane names to the displayed roles without altering match records', async () => {
  const matches = ['MIDDLE', 'BOTTOM', 'UTILITY'].map((role, index) => ({
    match_id: 'EUW_' + index, champion: 'Ahri', role, win: true,
    kills: 4, deaths: 2, assists: 8, kda: '6.0', cs: 180, cs_per_min: '6.0',
    gold: 10000, duration: '30:00', game_mode: 'CLASSIC', time_ago: 'Hier', items: [],
  }));
  const api = apiWith(async () => ok({ success: true, player: { name: 'Known', tag: 'EUW', has_data: true }, matches }));
  const result = await api.fetchPlayerHistory('Known');
  assert.deepEqual(Array.from(result.matches, (match) => match.role), ['MID', 'ADC', 'SUPPORT']);
  assert.equal(result.matches[0].kills, 4);
});
