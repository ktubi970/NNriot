import { test as base } from '@playwright/test';
import { DashboardPage } from '../pages/dashboard.page.js';
import { PredictorPage } from '../pages/predictor.page.js';

type CustomFixtures = {
  dashboardPage: DashboardPage;
  predictorPage: PredictorPage;
};

export const test = base.extend<CustomFixtures>({
  dashboardPage: async ({ page }, use) => {
    const dashboardPage = new DashboardPage(page);
    await use(dashboardPage);
  },
  predictorPage: async ({ page }, use) => {
    const predictorPage = new PredictorPage(page);
    await use(predictorPage);
  },
});

export { expect } from '@playwright/test';
