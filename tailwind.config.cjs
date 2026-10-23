const path = require('path');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    path.join(__dirname, 'frontend/index.html'),
    path.join(__dirname, 'frontend/src/**/*.{js,ts,jsx,tsx}'),
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#101114',
          surface: '#191b20',
          card: '#22252b',
          border: '#343840',
          cyan: '#80b5ff',
          neonPink: '#f4a1a7',
          neonAmber: '#b0b5bf',
          neonGreen: '#b0b5bf',
          neonPurple: '#b0b5bf',
          blueTeam: '#80b5ff',
          redTeam: '#f4a1a7',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
        sans: ['-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'cyber-card': '0 8px 24px rgba(0, 0, 0, 0.12)',
      },
    },
  },
  plugins: [],
};
