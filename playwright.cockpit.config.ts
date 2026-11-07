import { defineConfig } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Desktop sandbox sessions may remap LOCALAPPDATA away from the user's browser cache.
const browserCache = process.env.PLAYWRIGHT_BROWSERS_PATH || path.join(os.homedir(), 'AppData', 'Local', 'ms-playwright');
const chromium = fs.existsSync(browserCache)
  ? fs.readdirSync(browserCache)
      .filter(entry => /^chromium-\d+$/.test(entry))
      .sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]))
      .map(entry => path.join(browserCache, entry, 'chrome-win64', 'chrome.exe'))
      .find(file => fs.existsSync(file))
  : undefined;

export default defineConfig({
  testDir: './e2e',
  testMatch: 'cockpit_contract.spec.ts',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['html', { open: 'never', outputFolder: 'playwright-report' }], ['list']],
  outputDir: 'output/playwright/cockpit-contract',
  use: {
    baseURL: 'http://127.0.0.1:5175',
    browserName: 'chromium',
    launchOptions: chromium ? { executablePath: chromium } : {},
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    serviceWorkers: 'block',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'tablet', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'mobile-reduced-motion', use: { viewport: { width: 390, height: 844 }, contextOptions: { reducedMotion: 'reduce' } } },
  ],
  webServer: {
    command: 'node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5175 --strictPort',
    url: 'http://127.0.0.1:5175',
    reuseExistingServer: false,
    timeout: 30_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
