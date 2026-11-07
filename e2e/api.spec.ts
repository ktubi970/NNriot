import { test, expect } from '@playwright/test';

test.describe('NNriot Backend API', () => {
  test('GET /api/status should return API status and health info', async ({ request }) => {
    const response = await request.get('/api/status');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty('model_loaded');
    expect(data).toHaveProperty('status');
    expect(data).toHaveProperty('output_classes', 2);
  });

  test('POST /api/predict should reject invalid requests with 400', async ({ request }) => {
    const response = await request.post('/api/predict', {
      data: {
        blue_team: [],
        red_team: [],
      },
    });

    // An incomplete team lineup should return bad request
    expect([400, 422]).toContain(response.status());
  });
});
