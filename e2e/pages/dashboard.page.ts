import { Page, Locator, expect } from '@playwright/test';

export class DashboardPage {
  readonly page: Page;
  readonly logo: Locator;
  readonly dashboardNavLink: Locator;
  readonly predictorNavLink: Locator;
  readonly quickPredictBtn: Locator;
  readonly mobileMenuToggle: Locator;
  readonly mobileMenu: Locator;
  readonly heroTitle: Locator;
  readonly launchPredictorBtn: Locator;
  readonly accuracyCard: Locator;
  readonly matchesAnalyzedCard: Locator;
  readonly liveInferenceCard: Locator;
  readonly chartTitle: Locator;

  constructor(page: Page) {
    this.page = page;
    this.logo = page.locator('#main-nav a', { hasText: 'NN' });
    this.dashboardNavLink = page.locator('#main-nav').getByRole('link', { name: 'Dashboard' });
    this.predictorNavLink = page.locator('#main-nav').getByRole('link', { name: 'Predictor', exact: true });
    this.quickPredictBtn = page.getByRole('link', { name: /Quick Predict/i });
    this.mobileMenuToggle = page.locator('#menu-toggle');
    this.mobileMenu = page.locator('#mobile-menu');
    this.heroTitle = page.getByRole('heading', { level: 1, name: /AI Match Prediction/i });
    this.launchPredictorBtn = page.getByRole('link', { name: 'Launch Predictor' });
    this.accuracyCard = page.locator('.glass-panel', { hasText: 'Model Accuracy' });
    this.matchesAnalyzedCard = page.locator('.glass-panel', { hasText: 'Matches Analyzed' });
    this.liveInferenceCard = page.locator('.glass-panel', { hasText: 'Live Inference Time' });
    this.chartTitle = page.getByRole('heading', { level: 2, name: /Model Training History/i });
  }

  async goto() {
    await this.page.goto('/');
  }

  async clickLaunchPredictor() {
    await this.launchPredictorBtn.click();
    await expect(this.page).toHaveURL(/.*predictor/);
  }

  async toggleMobileMenu() {
    await this.mobileMenuToggle.click();
  }
}
