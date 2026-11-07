import { test, expect, Route } from '@playwright/test';
import { RAW_RESPONSE, CALIBRATED_RESPONSE, completeDraft, isolateNetwork, resultPanel, expectFocusWithin } from './fixtures/cockpit-contract';

test('a failed prediction exposes the error without inventing probabilities or kills', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({
    status: 503,
    json: { success: false, error: 'Modèle temporairement indisponible', error_code: 'model_contract' },
  }));
  await page.goto('/');
  const response = page.waitForResponse('**/api/predict/custom');
  await completeDraft(page);
  await response;
  const result = resultPanel(page);
  await expect(result).not.toContainText(/\d+(?:[.,]\d+)?\s*%/);
  await expect(result).not.toContainText(/KILLS ESTIMÉS|Durée attendue|Durée estimée/i);
  await expect(result.getByRole('alert')).toContainText(/indisponible/i);
});

test('uncalibrated output stays visibly raw and preserves the backend abstention', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: RAW_RESPONSE }));
  await page.goto('/');
  const response = page.waitForResponse('**/api/predict/custom');
  await completeDraft(page);
  await response;
  const result = resultPanel(page);
  await expect(result).toContainText(/62(?:[.,]0)?\s*%/);
  await expect(result).toContainText(/(?:brut|non calibr)/i);
  await expect(result).toContainText(/abstention/i);
  await expect(result).not.toContainText(/KILLS ESTIMÉS|Durée attendue|Durée estimée/i);
});

test('calibrated probability is primary while raw probability remains inspectable', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  const response = page.waitForResponse('**/api/predict/custom');
  await completeDraft(page);
  await response;
  const result = resultPanel(page);
  await expect(result.getByText(/^55\s*%$/).first()).toBeVisible();
  await expect(result.getByText(/^45\s*%$/).first()).toBeVisible();
  await expect(result).toContainText(/abstention/i);
  await expect(result.getByText('Conviction', { exact: true }).locator('..')).toContainText(/5\s*%/);
  await expect(result.getByText('Seuil appliqué', { exact: true }).locator('..')).toContainText(/10\s*%/);
  await result.getByText('Détails du résultat', { exact: true }).click();
  await expect(result.getByText(/62\s*%/).first()).toBeVisible();
  await expect(result).not.toContainText(/KILLS ESTIMÉS|Durée attendue|Durée estimée/i);
});

test('an incomplete draft shows an honest empty result and sends no prediction request', async ({ page }) => {
  const requests: unknown[] = [];
  await isolateNetwork(page, async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({ json: CALIBRATED_RESPONSE });
  });
  await page.goto('/');
  const result = resultPanel(page);
  await expect(result).toContainText(/Complétez les deux équipes/i);
  await expect(result).not.toContainText(/\d+(?:[.,]\d+)?\s*%/);
  await page.getByRole('button', { name: /Composer aléatoirement l’équipe bleue/i }).click();
  await expect(result).toContainText(/Complétez les deux équipes/i);
  // Wait beyond the inference debounce: a request from an incomplete draft is a side effect bug.
  await page.waitForTimeout(450);
  expect(requests).toHaveLength(0);
  await expect(page.getByText(/Saken|Upset|Vetheo|BrokenBlade|Hans Sama/)).toHaveCount(0);
});

test('champion selection is keyboard usable and its dialog traps and restores focus', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Choisir un champion · TOP · équipe bleue', exact: true });
  await opener.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Choisir un champion', exact: true });
  await expectFocusWithin(page, 'Choisir un champion');
  await expect(dialog.getByRole('textbox', { name: 'Rechercher un champion', exact: true })).toBeFocused();
  await dialog.locator('button:enabled, input:enabled, textarea:enabled, select:enabled, a[href], [tabindex="0"]').last().focus();
  await page.keyboard.press('Tab');
  await expectFocusWithin(page, 'Choisir un champion');
  await dialog.getByRole('button').first().focus();
  await page.keyboard.press('Shift+Tab');
  await expectFocusWithin(page, 'Choisir un champion');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();

  await page.keyboard.press('Enter');
  await dialog.getByRole('textbox', { name: 'Rechercher un champion', exact: true }).fill('Yasuo');
  const champion = dialog.getByRole('button', { name: /Yasuo/ });
  await champion.focus();
  await page.keyboard.press('Enter');
  await expect(dialog).toBeHidden();
  await expect(page.getByRole('button', { name: 'Changer Yasuo · TOP · équipe bleue', exact: true })).toBeVisible();
});

test('the tools drawer traps focus and Escape returns it to the trigger', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Analyse', exact: true });
  await opener.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Outils de la draft', exact: true });
  await expectFocusWithin(page, 'Outils de la draft');
  await dialog.locator('button:enabled, input:enabled, textarea:enabled, select:enabled, a[href], [tabindex="0"]').last().focus();
  await page.keyboard.press('Tab');
  await expectFocusWithin(page, 'Outils de la draft');
  await dialog.getByRole('button').first().focus();
  await page.keyboard.press('Shift+Tab');
  await expectFocusWithin(page, 'Outils de la draft');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test('closing player search restores focus to the player association button', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Associer un joueur · TOP · équipe bleue', exact: true });
  await opener.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Choisir un joueur', exact: true });
  await expect(dialog.getByRole('combobox', { name: 'Rechercher un joueur', exact: true })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test('the history tools entry has a selected tab and reachable keyboard navigation', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Historique', exact: true });
  await opener.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Outils de la draft', exact: true });
  const historyTab = dialog.getByRole('tab', { name: 'Historique', exact: true });
  await expect(historyTab).toHaveAttribute('aria-selected', 'true');
  await expect(dialog.getByRole('button', { name: 'Fermer les outils', exact: true })).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(historyTab).toBeFocused();
  await expect(dialog.getByRole('tabpanel', { name: 'Historique', exact: true })).toContainText('Sélectionnez un joueur');
  await page.keyboard.press('ArrowRight');
  const analyticsTab = dialog.getByRole('tab', { name: 'Analyse', exact: true });
  await expect(analyticsTab).toBeFocused();
  await expect(analyticsTab).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('End');
  await expect(historyTab).toBeFocused();
  await expect(historyTab).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test('closing a player history dialog restores focus after its exit animation starts', async ({ page }) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.route('**/api/players/search?*', route => route.fulfill({ json: [{
    game_name: 'Test Player', tag_line: 'EUW', puuid: 'stored-test-player',
    total_matches: 0, winrate: 0, avg_kda: 0,
  }] }));
  await page.route('**/api/player/history?*', route => route.fulfill({ json: {
    success: true, player: { game_name: 'Test Player', tag_line: 'EUW', total_games: 0, avg_kda: 0 }, matches: [],
  } }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Associer un joueur · TOP · équipe bleue', exact: true }).click();
  const searchDialog = page.getByRole('dialog', { name: 'Choisir un joueur', exact: true });
  await searchDialog.getByRole('combobox', { name: 'Rechercher un joueur', exact: true }).fill('Test');
  await expect(searchDialog.getByRole('option', { name: /Test Player#EUW/ })).toBeVisible();
  await page.keyboard.press('Enter');
  await expect(searchDialog).toBeHidden();
  await expect(page.getByRole('button', { name: 'Changer le joueur · TOP · équipe bleue', exact: true })).toBeFocused();
  const opener = page.getByRole('button', { name: 'Voir l’historique de Test Player', exact: true });
  await opener.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Historique de Test Player', exact: true });
  await expect(dialog.getByRole('button', { name: 'Fermer l’historique', exact: true })).toBeFocused();
  await expect(dialog.getByText('Aucun match enregistré pour ce profil.', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test('a delayed response cannot replace the result for the current teams', async ({ page }) => {
  let delayed: Route | undefined;
  let requests = 0;
  await isolateNetwork(page, async route => {
    requests += 1;
    if (requests === 1) { delayed = route; return; }
    await route.fulfill({ json: {
      ...CALIBRATED_RESPONSE,
      nn_results: { blue_win_probability: 0.77, red_win_probability: 0.23 },
      predicted_winner: 'TEAM_A', confidence: 0.77,
      blue_win_probability: 0.77, red_win_probability: 0.23,
      blue_win_prob: 0.77, red_win_prob: 0.23,
      probabilities_calibrated: { team_a: 0.77, team_b: 0.23 },
      selective_decision: { ...CALIBRATED_RESPONSE.selective_decision, conviction: 0.27, abstained: false, is_actionable: true, tier: 'HIGH' },
      multi_output: {
        winner: { team_a: 0.77, team_b: 0.23, predicted: 'team_a', confidence: 0.77 },
        selective_decision: { ...CALIBRATED_RESPONSE.selective_decision, conviction: 0.27, abstained: false, is_actionable: true, tier: 'HIGH' },
      },
    } });
  });
  await page.goto('/');
  await completeDraft(page);
  await expect.poll(() => Boolean(delayed)).toBe(true);
  const latest = page.waitForResponse('**/api/predict/custom');
  await page.getByRole('button', { name: 'Inverser les équipes', exact: true }).click();
  await latest;
  const result = resultPanel(page);
  await expect(result.getByText(/^77\s*%$/).first()).toBeVisible();
  await delayed!.fulfill({ json: CALIBRATED_RESPONSE }).catch(() => { /* Fetch may already be aborted. */ });
  await page.waitForTimeout(250);
  await expect(result.getByText(/^77\s*%$/).first()).toBeVisible();
  await expect(result.getByText(/^55\s*%$/)).toHaveCount(0);
});

test('draft controls and results remain reachable without horizontal clipping', async ({ page }, testInfo) => {
  await isolateNetwork(page, route => route.fulfill({ json: CALIBRATED_RESPONSE }));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Analyse de draft', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Équipe bleue', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Équipe rouge', exact: true })).toBeVisible();
  await expect(resultPanel(page)).toBeVisible();
  await expect(page.getByRole('region', { name: 'Contributions', exact: true })).toBeVisible();
  const documentFits = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
  expect(documentFits).toBe(true);
  await page.screenshot({ path: `output/playwright/cockpit-empty-${testInfo.project.name}.png`, fullPage: true });
  const response = page.waitForResponse('**/api/predict/custom');
  await completeDraft(page);
  await response;
  await expect(resultPanel(page).getByText(/^55\s*%$/).first()).toBeVisible();
  await page.screenshot({ path: `output/playwright/cockpit-${testInfo.project.name}.png`, fullPage: true });
});
