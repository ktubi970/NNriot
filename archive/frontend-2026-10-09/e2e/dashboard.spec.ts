import { test, expect } from './fixtures/test-fixtures.js';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ dashboardPage }) => {
    await dashboardPage.goto();
  });

  test('should load successfully with title and navigation', async ({ page, dashboardPage, isMobile }) => {
    await expect(page).toHaveTitle(/Dashboard – NNriot/);
    await expect(dashboardPage.logo).toBeVisible();
    if (isMobile) {
      await expect(dashboardPage.mobileMenuToggle).toBeVisible();
    } else {
      await expect(dashboardPage.dashboardNavLink).toBeVisible();
      await expect(dashboardPage.predictorNavLink).toBeVisible();
    }
  });

  test('should display hero section with CTA leading to predictor', async ({ dashboardPage }) => {
    await expect(dashboardPage.heroTitle).toBeVisible();
    await expect(dashboardPage.launchPredictorBtn).toBeVisible();
    await dashboardPage.clickLaunchPredictor();
  });

  test('should display AI model performance stats cards', async ({ dashboardPage }) => {
    await expect(dashboardPage.accuracyCard).toBeVisible();
    await expect(dashboardPage.matchesAnalyzedCard).toBeVisible();
    await expect(dashboardPage.liveInferenceCard).toBeVisible();
    await expect(dashboardPage.chartTitle).toBeVisible();
  });

  test('should toggle mobile menu on mobile viewport', async ({ page, dashboardPage }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(dashboardPage.mobileMenuToggle).toBeVisible();
    await expect(dashboardPage.mobileMenu).toBeHidden();

    await dashboardPage.toggleMobileMenu();
    await expect(dashboardPage.mobileMenu).toBeVisible();

    await dashboardPage.toggleMobileMenu();
    await expect(dashboardPage.mobileMenu).toBeHidden();
  });
});
