import { Page, Locator, expect } from '@playwright/test';

export class PredictorPage {
  readonly page: Page;
  readonly heading: Locator;
  readonly randomTeamsBtn: Locator;
  readonly swapTeamsBtn: Locator;
  readonly clearAllBtn: Locator;
  readonly popularPlayersSelect: Locator;
  readonly blueTeamHeader: Locator;
  readonly redTeamHeader: Locator;
  readonly blueSlots: Locator;
  readonly redSlots: Locator;
  readonly predictBtn: Locator;
  readonly loader: Locator;
  readonly resultsContainer: Locator;
  readonly winnerText: Locator;
  readonly labelBlue: Locator;
  readonly labelRed: Locator;
  readonly confidenceVal: Locator;
  readonly marketsContainer: Locator;
  readonly marketTotalKills: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading = page.getByRole('heading', { level: 1, name: /CUSTOM MATCH PREDICTOR/i });
    this.randomTeamsBtn = page.getByRole('button', { name: /Random Teams/i });
    this.swapTeamsBtn = page.getByRole('button', { name: /Swap Teams/i });
    this.clearAllBtn = page.getByRole('button', { name: /Clear All/i });
    this.popularPlayersSelect = page.locator('#hot-players-select');
    this.blueTeamHeader = page.locator('.team-card.blue h2');
    this.redTeamHeader = page.locator('.team-card.red h2');
    this.blueSlots = page.locator('#blue-slots .player-slot');
    this.redSlots = page.locator('#red-slots .player-slot');
    this.predictBtn = page.locator('#predict-btn');
    this.loader = page.locator('#loader');
    this.resultsContainer = page.locator('#results-container');
    this.winnerText = page.locator('#winner-text');
    this.labelBlue = page.locator('#label-blue');
    this.labelRed = page.locator('#label-red');
    this.confidenceVal = page.locator('#confidence-val');
    this.marketsContainer = page.locator('#markets-container');
    this.marketTotalKills = page.locator('#market-total-kills');
  }

  async goto() {
    await this.page.goto('/predictor');
  }

  async mockRandomPlayers(playersCount: number = 10) {
    const players = Array.from({ length: playersCount }, (_, i) => ({
      puuid: `puuid-mock-${i + 1}`,
      game_name: `ProPlayer${i + 1}`,
      tag_line: `EUW${i + 1}`,
      total_matches: 42 + i,
    }));

    await this.page.route(/\/api\/players\/random/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(players),
      });
    });
  }

  async mockPredictionResponse(customResponse?: Record<string, unknown>) {
    const defaultResponse = {
      success: true,
      blue_win_probability: 0.68,
      red_win_probability: 0.32,
      confidence: 0.85,
      predicted_winner: 'BLUE',
      explanation: 'Dominant early game lineup with high gold advantage probability.',
      multi_output: {
        winner: { team_a: 0.68, team_b: 0.32, predicted: 'A', confidence: 0.85 },
        team_b_kill_lead: { team_a: 0.72, team_b: 0.28, predicted: 'A' },
        kills: {
          total: 34.5,
          team_a: 21.0,
          team_b: 13.5,
          handicap: 7.5,
          odd_probability: 0.45,
        },
        first: {
          blood: { team_a: 0.65, team_b: 0.35, none: 0.0 },
          baron: { team_a: 0.70, team_b: 0.25, none: 0.05 },
          inhibitor: { team_a: 0.72, team_b: 0.23, none: 0.05 },
          tower: { team_a: 0.60, team_b: 0.38, none: 0.02 },
          kills_5: { team_a: 0.62, team_b: 0.35, none: 0.03 },
          kills_10: { team_a: 0.65, team_b: 0.32, none: 0.03 },
          kills_15: { team_a: 0.68, team_b: 0.29, none: 0.03 },
          kills_20: { team_a: 0.70, team_b: 0.27, none: 0.03 },
        },
        totals: {
          barons: 1.5,
          dragons: 4.2,
          towers: 12.0,
        },
        both_teams: {
          baron: 0.20,
          inhibitor: 0.25,
          dragon: 0.80,
        },
        elder_dragon: 0.22,
      },
    };

    await this.page.route(/\/api\/predict/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(customResponse || defaultResponse),
      });
    });
  }

  async fillRandomTeams() {
    await this.randomTeamsBtn.click();
  }

  async swapTeams() {
    await this.swapTeamsBtn.click();
  }

  async clearAll() {
    await this.clearAllBtn.click();
  }

  async clickPredict() {
    await this.predictBtn.click();
  }
}
