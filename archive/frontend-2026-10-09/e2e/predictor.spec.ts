import { test, expect } from './fixtures/test-fixtures.js';

test.describe('Custom Match Predictor Page', () => {
  test.beforeEach(async ({ predictorPage }) => {
    await predictorPage.goto();
  });

  test('should render 10 player slots and disabled predict button initially', async ({ predictorPage }) => {
    await expect(predictorPage.heading).toBeVisible();
    await expect(predictorPage.blueTeamHeader).toHaveText(/BLUE TEAM/i);
    await expect(predictorPage.redTeamHeader).toHaveText(/RED TEAM/i);

    await expect(predictorPage.blueSlots).toHaveCount(5);
    await expect(predictorPage.redSlots).toHaveCount(5);
    await expect(predictorPage.predictBtn).toBeDisabled();
  });

  test('should populate all 10 slots when clicking Random Teams', async ({ predictorPage, page }) => {
    await predictorPage.mockRandomPlayers(10);
    await predictorPage.fillRandomTeams();

    // Verify all 10 slots now display selected player components
    const selectedPlayers = page.locator('.selected-player:visible');
    await expect(selectedPlayers).toHaveCount(10);

    // Predict button should now be enabled
    await expect(predictorPage.predictBtn).toBeEnabled();
  });

  test('should clear all slots when clicking Clear All', async ({ predictorPage, page }) => {
    await predictorPage.mockRandomPlayers(10);
    await predictorPage.fillRandomTeams();
    await expect(predictorPage.predictBtn).toBeEnabled();

    await predictorPage.clearAll();

    const selectedPlayers = page.locator('.selected-player:visible');
    await expect(selectedPlayers).toHaveCount(0);
    await expect(predictorPage.predictBtn).toBeDisabled();
  });

  test('should swap Blue and Red teams when clicking Swap Teams', async ({ predictorPage, page }) => {
    await predictorPage.mockRandomPlayers(10);
    await predictorPage.fillRandomTeams();

    // Ensure players have populated before capturing names
    await expect(page.locator('#name-blue-0')).not.toHaveText('-');
    await expect(page.locator('#name-red-0')).not.toHaveText('-');

    const firstBluePlayerName = (await page.locator('#name-blue-0').textContent())?.trim() || '';
    const firstRedPlayerName = (await page.locator('#name-red-0').textContent())?.trim() || '';

    await predictorPage.swapTeams();

    await expect(page.locator('#name-blue-0')).toHaveText(firstRedPlayerName);
    await expect(page.locator('#name-red-0')).toHaveText(firstBluePlayerName);
  });

  test('should execute prediction and display win probabilities and multi-output markets', async ({ predictorPage, page }) => {
    await predictorPage.mockRandomPlayers(10);
    await predictorPage.mockPredictionResponse();

    await predictorPage.fillRandomTeams();
    await predictorPage.clickPredict();

    // Results container should be displayed with probability breakdown
    await expect(predictorPage.resultsContainer).toBeVisible();
    await expect(predictorPage.winnerText).toHaveText(/VICTORY/i);
    await expect(predictorPage.labelBlue).toHaveText(/68(\.0)?%/);
    await expect(predictorPage.labelRed).toHaveText(/32(\.0)?%/);

    // Multi-output markets container should be rendered
    await expect(predictorPage.marketsContainer).toBeVisible();
    await expect(predictorPage.marketTotalKills).toHaveText(/34\.5/);
    await expect(page.locator('#market-team-a-kills')).toHaveText(/21\.0/);
    await expect(page.locator('#market-team-b-kills')).toHaveText(/13\.5/);
  });
});
