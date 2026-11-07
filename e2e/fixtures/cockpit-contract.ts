import { expect, Page, Route } from '@playwright/test';

const uncalibratedDecision = {
  conviction: 0.12,
  tier: 'UNCALIBRATED',
  is_actionable: false,
  threshold_applied: 0.1,
  high_threshold: 0.2,
  abstained: true,
  recommendation: 'Abstain - winner calibration unavailable',
};

export const RAW_RESPONSE = {
  success: true,
  nn_results: { blue_win_probability: 0.62, red_win_probability: 0.38 },
  predicted_winner: 'ABSTAIN',
  confidence: 0.62,
  blue_win_probability: 0.62,
  red_win_probability: 0.38,
  blue_win_prob: 0.62,
  red_win_prob: 0.38,
  explanation: 'Multi-head NN analysis complete (PredictionService).',
  llm_active: false,
  model: {
    id: 'contract-fixture',
    checkpoint_sha256: null,
    feature_contract: 'draft-v4',
    legacy_architecture: false,
    training_provenance_status: 'verified',
    serving_validation: { status: 'validated' },
  },
  probabilities_raw: { team_a: 0.62, team_b: 0.38 },
  probabilities_calibrated: null,
  calibration: { status: 'unavailable', temperature: null },
  draft_advantage: null,
  priors_provenance: { policy: 'declared_cold_start_defaults', history_available: false },
  selective_decision: uncalibratedDecision,
  shapley_credits: null,
  draft_composition: null,
  epistemic_uncertainty: null,
  multi_output: {
    winner: { team_a: 0.62, team_b: 0.38, predicted: 'abstain', confidence: 0.62 },
    selective_decision: uncalibratedDecision,
  },
};

const calibratedDecision = {
  ...uncalibratedDecision,
  conviction: 0.05,
  tier: 'ABSTAIN',
  recommendation: 'Abstain - conviction below threshold',
};

export const CALIBRATED_RESPONSE = {
  ...RAW_RESPONSE,
  nn_results: { blue_win_probability: 0.55, red_win_probability: 0.45 },
  confidence: 0.55,
  blue_win_probability: 0.55,
  red_win_probability: 0.45,
  blue_win_prob: 0.55,
  red_win_prob: 0.45,
  probabilities_calibrated: { team_a: 0.55, team_b: 0.45 },
  calibration: { status: 'artifact_verified', temperature: 2.5 },
  selective_decision: calibratedDecision,
  multi_output: {
    winner: { team_a: 0.55, team_b: 0.45, predicted: 'abstain', confidence: 0.55 },
    selective_decision: calibratedDecision,
  },
};

const names = [
  ['Aatrox', '266', ['Fighter']], ['Renekton', '58', ['Fighter']],
  ['LeeSin', '64', ['Fighter']], ['Sejuani', '113', ['Tank']],
  ['Ahri', '103', ['Mage']], ['Sylas', '517', ['Mage']],
  ['Jinx', '222', ['Marksman']], ['Ezreal', '81', ['Marksman']],
  ['Thresh', '412', ['Support']], ['Leona', '89', ['Support']],
  ['Yasuo', '157', ['Fighter', 'Assassin']],
] as const;

const catalogue = {
  type: 'champion', format: 'standAloneComplex', version: '14.24.1',
  data: Object.fromEntries(names.map(([id, key, tags]) => [id, {
    id, key, name: id === 'LeeSin' ? 'Lee Sin' : id, title: 'Catalogue fixture',
    blurb: 'Deterministic champion metadata for UI contracts.',
    info: { attack: 5, defense: 5, magic: 5, difficulty: 5 },
    tags, partype: 'Mana',
    image: { full: `${id}.png`, sprite: 'champion0.png', group: 'champion', x: 0, y: 0, w: 48, h: 48 },
    stats: { hp: 600, hpperlevel: 100, mp: 300, mpperlevel: 50, movespeed: 335, armor: 30, armorperlevel: 4, spellblock: 30, spellblockperlevel: 1.3, attackrange: 175, hpregen: 5, hpregenperlevel: 0.5, mpregen: 7, mpregenperlevel: 0.7, crit: 0, critperlevel: 0, attackdamage: 60, attackdamageperlevel: 3, attackspeedperlevel: 2, attackspeed: 0.65 },
  }])),
};
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6eB8AAAAASUVORK5CYII=', 'base64');

export async function isolateNetwork(page: Page, prediction: (route: Route) => Promise<void>) {
  // Keep the real React components, requests, adapters and hooks. Replace only HTTP boundaries.
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/predict/custom') return prediction(route);
    if (pathname === '/api/status') return route.fulfill({ json: {
      model_loaded: true, tensorflow_version: '2.20', input_dimension: 18,
      output_classes: 2, status: 'ready',
    } });
    if (/players\/search|lolpros\/search/.test(pathname)) return route.fulfill({ json: [] });
    if (pathname === '/api/database/stats') return route.fulfill({ json: {
      total_matches: 0, total_training: 0, untrained_records: 0, total_players: 0,
      crawl_queue_pending: 0, total_training_rows: 0, untrained_training_rows: 0,
    } });
    if (/history/.test(pathname)) return route.fulfill({ json: { player: null, matches: [] } });
    return route.fulfill({ status: 503, json: { success: false, error: 'Fixture: endpoint unavailable' } });
  });
  await page.route('**/analytics/**', route => route.fulfill({ status: 503, json: { error: 'Fixture: analytics unavailable' } }));
  await page.route('https://ddragon.leagueoflegends.com/**', route => route.request().url().endsWith('.json')
    ? route.fulfill({ json: catalogue })
    : route.fulfill({ contentType: 'image/png', body: pixel }));
}

export async function completeDraft(page: Page) {
  const compose = page.getByRole('button', { name: /Composer aléatoirement l’équipe/i });
  await expect(compose).toHaveCount(2);
  await compose.first().click();
  await compose.last().click();
}

export function resultPanel(page: Page) {
  return page.getByRole('region', { name: 'Résultat', exact: true });
}

export async function expectFocusWithin(page: Page, dialogName: string) {
  const dialog = page.getByRole('dialog', { name: dialogName, exact: true });
  await expect(dialog).toBeVisible();
  await expect.poll(() => dialog.evaluate(element => element.contains(document.activeElement))).toBe(true);
}
