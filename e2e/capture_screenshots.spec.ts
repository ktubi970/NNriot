import { test } from '@playwright/test';
import path from 'path';

const ARTIFACTS_DIR = 'C:/Users/ktubi/.gemini/antigravity/brain/33a2e9ae-909c-43c5-b21f-f95a4a78c3b9';

test('Capture Live Screenshots of the Cyber Analytics Cockpit', async ({ page }) => {
  // Set high resolution viewport
  await page.setViewportSize({ width: 1440, height: 900 });

  // 1. Panoramic Cockpit Main View
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'cockpit_main_view.png'), fullPage: false });

  // 1b. Player Autocompletion Dropdown Open
  const playerChip = page.locator('text=Vetheo').first();
  await playerChip.click();
  await page.waitForTimeout(300);
  const autoInput = page.locator('input[placeholder*="Chercher joueur"]');
  await autoInput.fill('Fa');
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'cockpit_player_autocomplete.png'), fullPage: false });
  await autoInput.press('Enter');
  await page.waitForTimeout(500);

  // 2. Champion Picker Modal Open
  const firstSlot = page.locator('text=Aatrox').first();
  await firstSlot.click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'cockpit_champion_picker.png'), fullPage: false });

  // Close modal by selecting Yasuo
  await page.locator('button:has-text("Yasuo")').click();
  await page.waitForTimeout(500);

  // 3. Cyber Drawer - Analytics Chat AI with Evidence
  const analyticsBtn = page.locator('button:has-text("ANALYTICS AI")');
  await analyticsBtn.click();
  await page.waitForTimeout(500);
  const samplePrompt = page.locator('button:has-text("Impact du premier Dragon")');
  await samplePrompt.click();
  await page.waitForSelector('text=PREUVE STATISTIQUE', { timeout: 10000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'cockpit_drawer_chat.png'), fullPage: false });

  // 4. Cyber Drawer - Spectator Live & OCR Stream
  const spectatorTab = page.getByRole('button', { name: 'SPECTATOR', exact: true });
  await spectatorTab.click();
  await page.waitForTimeout(500);
  const ocrBtn = page.locator('button:has-text("OCR FLUX")');
  await ocrBtn.click();
  await page.waitForSelector('text=STREAM_OCR_LIVE_01', { timeout: 5000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'cockpit_drawer_spectator.png'), fullPage: false });
});
