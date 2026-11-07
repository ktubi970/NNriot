import { defineConfig, devices } from '@playwright/test';
import fs from 'fs';
import path from 'path';

function getPythonCommand(): string {
  if (process.env.PYTHON) return `"${process.env.PYTHON}" cli.py web`;
  if (process.platform === 'win32') {
    const venv312 = path.resolve(__dirname, '.venv312', 'Scripts', 'python.exe');
    if (fs.existsSync(venv312)) return `"${venv312}" cli.py web`;
    const venv = path.resolve(__dirname, '.venv', 'Scripts', 'python.exe');
    if (fs.existsSync(venv)) return `"${venv}" cli.py web`;
  } else {
    const venv = path.resolve(__dirname, '.venv', 'bin', 'python');
    if (fs.existsSync(venv)) return `"${venv}" cli.py web`;
  }
  return 'python cli.py web';
}

/**
 * Playwright Pro Configuration for NNriot Web App
 * See https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: './e2e',
  /* Maximum time one test can run for. */
  timeout: 30 * 1000,
  expect: {
    /**
     * Maximum time expect() should wait for the condition to be met.
     */
    timeout: 5000
  },
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI to avoid overloading backend services. */
  workers: process.env.CI ? 2 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html', { open: 'never' }],
    ['list']
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.BASE_URL || 'http://127.0.0.1:5000',

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    /* Mobile viewports testing */
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],

  /* Run your local dev server before starting the tests */
  webServer: {
    command: getPythonCommand(),
    url: 'http://127.0.0.1:5000',
    reuseExistingServer: !process.env.CI,
    timeout: 60 * 1000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
