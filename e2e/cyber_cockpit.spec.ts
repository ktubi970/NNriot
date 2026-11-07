import { test, expect } from '@playwright/test';

test.describe('NNriot Cyber Analytics Cockpit - Full Interactive Live Test Suite', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to Cockpit
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
  });

  test('1. Cockpit Header & Telemetry Initialization', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/NNriot \| Cyber Analytics Cockpit/);

    // Check Header elements
    const brandTitle = page.locator('h1:has-text("NNriot")');
    await expect(brandTitle).toBeVisible();

    const versionBadge = page.locator('text=v2.0 CYBER COCKPIT');
    await expect(versionBadge).toBeVisible();

    // Check Telemetry Strip
    await expect(page.locator('text=MODEL:')).toBeVisible();
    await expect(page.locator('text=VECTOR:')).toBeVisible();
    await expect(page.locator('text=PING:')).toBeVisible();

    // Test Region selector dropdown
    const regionSelect = page.locator('header select');
    await expect(regionSelect).toBeVisible();
    await regionSelect.selectOption('KR');
    await expect(regionSelect).toHaveValue('KR');
    await regionSelect.selectOption('EUW');
    await expect(regionSelect).toHaveValue('EUW');
  });

  test('2. 5v5 Lineups, Champion Slots & Initial Roster', async ({ page }) => {
    // Check Blue Team & Red Team headers
    await expect(page.locator('h3:has-text("EQUIPE BLEUE")')).toBeVisible();
    await expect(page.locator('h3:has-text("EQUIPE ROUGE")')).toBeVisible();

    // Check 5 roles on both sides
    const roles = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'];
    for (const role of roles) {
      await expect(page.locator(`text=${role}`).first()).toBeVisible();
    }

    // Verify initial champions rendered in slots
    await expect(page.locator('text=Aatrox').first()).toBeVisible();
    await expect(page.locator('text=Renekton').first()).toBeVisible();
  });

  test('3. Champion Picker Modal, Role Filters, Search & Selection', async ({ page }) => {
    // Click on the first Blue TOP champion slot to open picker
    const firstChampSlot = page.locator('text=Aatrox').first();
    await firstChampSlot.click();

    // Verify Modal is opened
    const modalTitle = page.locator('h2:has-text("SELECT CHAMPION")');
    await expect(modalTitle).toBeVisible();

    // Test Role Filter buttons in modal
    const midFilter = page.locator('button:has-text("MID")').filter({ has: page.locator('svg') });
    await midFilter.click();

    const adcFilter = page.locator('button:has-text("ADC")').filter({ has: page.locator('svg') });
    await adcFilter.click();

    const allFilter = page.locator('button:has-text("ALL")');
    await allFilter.click();

    // Test search filter
    const searchInput = page.locator('input[placeholder*="Search champion"]');
    await searchInput.fill('Yasuo');
    await expect(page.locator('text=Yasuo')).toBeVisible();

    // Select Yasuo
    await page.locator('button:has-text("Yasuo")').click();

    // Modal should close and Yasuo should now be in the Blue Team
    await expect(modalTitle).toBeHidden();
    await expect(page.locator('text=Yasuo').first()).toBeVisible();
  });

  test('4. Player Summoner Autocompletion & Pro Search', async ({ page }) => {
    // Click player name or "Associer Joueur" on Blue MID slot
    const playerChip = page.locator('text=Vetheo').first();
    if (await playerChip.isVisible()) {
      await playerChip.click();
    } else {
      await page.locator('text=Associer Joueur').first().click();
    }

    // Verify Player Autocomplete Popover appears
    const autocompleteInput = page.locator('input[placeholder*="Chercher joueur"]');
    await expect(autocompleteInput).toBeVisible();

    // Type Faker and press Enter to select top match
    await autocompleteInput.fill('Faker');
    await page.waitForTimeout(300);
    await autocompleteInput.press('Enter');

    // Verify player is updated with Faker and T1 badge
    await expect(page.locator('text=Faker').first()).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=T1').first()).toBeVisible({ timeout: 5000 });
  });

  test('5. Lineup Action Buttons: Randomize, Reset, and Team Swap', async ({ page }) => {
    // Click Swap Blue / Red
    const swapBtn = page.locator('button:has-text("SWAP BLUE / RED")');
    await expect(swapBtn).toBeVisible();
    await swapBtn.click();

    // Click Randomize on Blue Team
    const randomBtns = page.locator('button[title*="Randomize"]');
    await randomBtns.first().click();

    // Click Reset on Red Team
    const resetBtns = page.locator('button[title*="Reset"]');
    await resetBtns.last().click();

    // Verify empty slot prompt appears
    await expect(page.locator('text=Pick Champion...').first()).toBeVisible();
  });

  test('6. Neural Winrate Gauge & SHAP Feature Matrix', async ({ page }) => {
    // Verify Neural Engine Central Card
    await expect(page.locator('text=NEURAL ENGINE INFERENCE')).toBeVisible();
    await expect(page.locator('text=VICTOIRE BLUE')).toBeVisible();

    // Verify Metrics (Confiance, Kills, Durée)
    await expect(page.locator('text=CONFIANCE')).toBeVisible();
    await expect(page.locator('text=KILLS ESTIMÉS')).toBeVisible();
    await expect(page.locator('text=DURÉE')).toBeVisible();

    // Verify SHAP Feature Matrix lower card
    await expect(page.locator('text=MATRICE D\'EXPLICABILITÉ IA (SHAP)')).toBeVisible();
    await expect(page.locator('text=VECTOR 512D')).toBeVisible();
  });

  test('7. Cyber Drawer: Analytics AI Chat', async ({ page }) => {
    // Open Analytics AI tab in header
    const analyticsBtn = page.locator('button:has-text("ANALYTICS AI")');
    await analyticsBtn.click();

    // Drawer should open
    await expect(page.locator('text=ANALYTICS EXPLAINABILITY MART')).toBeVisible();

    // Click a sample suggested question
    const samplePrompt = page.locator('button:has-text("Impact du premier Dragon")');
    await expect(samplePrompt).toBeVisible();
    await samplePrompt.click();

    // Check that question and response appear
    await expect(page.locator('p:has-text("Impact du premier Dragon")').first()).toBeVisible();
    await expect(page.locator('text=PREUVE STATISTIQUE').first()).toBeVisible({ timeout: 10000 });

    // Send custom question
    const chatInput = page.locator('input[placeholder*="Posez une question"]');
    await chatInput.fill('Est-ce que pick Aatrox top est avantageux ?');
    await page.locator('button:has-text("ENVOYER")').click();

    await expect(page.locator('p:has-text("Est-ce que pick Aatrox top est avantageux ?")').first()).toBeVisible();
    await expect(page.locator('text=PREUVE STATISTIQUE').nth(1)).toBeVisible({ timeout: 10000 });

    // Close drawer
    const closeBtn = page.locator('button[aria-label="Close drawer"]');
    await closeBtn.click();
    await expect(page.locator('text=ANALYTICS EXPLAINABILITY MART')).toBeHidden();
  });

  test('8. Cyber Drawer: Spectator Live & Stream OCR Import', async ({ page }) => {
    // Open Spectator Tab
    const spectatorBtn = page.locator('button:has-text("SPECTATOR LIVE")');
    await spectatorBtn.click();

    await expect(page.locator('text=SPECTATOR & STREAM OCR DETECTOR')).toBeVisible();

    // Test OCR Stream scan
    const ocrBtn = page.locator('button:has-text("OCR FLUX")');
    await ocrBtn.click();

    // Check detected match card
    await expect(page.locator('text=STREAM_OCR_LIVE_01')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('span:has-text("EQUIPE BLEUE")')).toBeVisible();

    // Click "CHARGER CE MATCH DANS LE COCKPIT 5V5"
    const importBtn = page.locator('button:has-text("CHARGER CE MATCH DANS LE COCKPIT 5V5")');
    await importBtn.click();

    // Drawer should close and cockpit should contain imported champions
    await expect(page.locator('text=SPECTATOR & STREAM OCR DETECTOR')).toBeHidden();
    await expect(page.locator('text=Viego').first()).toBeVisible();
  });

  test('9. Cyber Drawer: Model Continuous Trainer', async ({ page }) => {
    // Open Trainer Tab
    const trainerBtn = page.locator('button:has-text("TRAINER")');
    await trainerBtn.click();

    await expect(page.locator('text=NEURAL NETWORK ENGINE & CONTINUOUS TRAINER')).toBeVisible();
    await expect(page.locator('text=Deep ResNet (Skip-Connections)')).toBeVisible();

    // Click Manual Step button
    const stepBtn = page.locator('button:has-text("STEP MANUEL")');
    await stepBtn.click();

    // Verify status update
    await expect(page.locator('text=Succès').or(page.locator('text=Information')).or(page.locator('text=Step terminé'))).toBeVisible({ timeout: 10000 });

    // Close drawer
    const closeBtn = page.locator('button[aria-label="Close drawer"]');
    await closeBtn.click();
    await expect(page.locator('text=NEURAL NETWORK ENGINE & CONTINUOUS TRAINER')).toBeHidden();
  });
});
