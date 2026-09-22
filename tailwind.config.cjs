/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./frontend/index.html",
    "./frontend/src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#050811',
          surface: '#0b1120',
          card: '#0f172a',
          border: '#1e293b',
          cyan: '#00f0ff',
          neonPink: '#ff0055',
          neonAmber: '#ffb700',
          neonGreen: '#00ff88',
          neonPurple: '#a855f7',
          blueTeam: '#0ea5e9',
          redTeam: '#f43f5e',
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      boxShadow: {
        'neon-cyan': '0 0 15px -3px rgba(0, 240, 255, 0.45)',
        'neon-pink': '0 0 15px -3px rgba(255, 0, 85, 0.45)',
        'neon-green': '0 0 15px -3px rgba(0, 255, 136, 0.45)',
        'cyber-card': '0 4px 20px -2px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.05)',
      },
      animation: {
        'pulse-glow': 'pulseGlow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scanline': 'scanline 8s linear infinite',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { opacity: '1', filter: 'drop-shadow(0 0 8px rgba(0, 240, 255, 0.6))' },
          '50%': { opacity: '0.6', filter: 'drop-shadow(0 0 2px rgba(0, 240, 255, 0.2))' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(1000%)' },
        }
      }
    },
  },
  plugins: [],
};
