import React, { useState } from 'react';
import { CyberBadge } from '../atoms/CyberBadge';
import { GlowButton } from '../atoms/GlowButton';
import { Eye, Radio, Search, Camera, ArrowRight, CheckCircle, AlertTriangle } from 'lucide-react';

interface LiveSpectatorTabProps {
  onImportLineup?: (blueChampions: string[], redChampions: string[]) => void;
}

export const LiveSpectatorTab: React.FC<LiveSpectatorTabProps> = ({ onImportLineup }) => {
  const [summonerQuery, setSummonerQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [liveData, setLiveData] = useState<any | null>(null);
  const [ocrActive, setOcrActive] = useState(false);

  const handleSearchSpectator = async () => {
    if (!summonerQuery.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/spectator?summoner=${encodeURIComponent(summonerQuery)}`);
      if (res.ok) {
        const data = await res.json();
        setLiveData(data);
      } else {
        // Mock live match for testing
        setLiveData({
          gameId: 'EUW1_712984102',
          gameMode: 'RANKED_SOLO_5x5',
          gameLength: 842,
          blue: ['Aatrox', 'LeeSin', 'Ahri', 'Jinx', 'Thresh'],
          red: ['Renekton', 'Sejuani', 'Sylas', 'Ezreal', 'Leona'],
          bluePlayers: ['KCorp Caliste', 'Bo', 'Vetheo', 'Upset', 'Targamas'],
          redPlayers: ['BrokenBlade', 'Yike', 'Caps', 'Hans Sama', 'Mikyx'],
        });
      }
    } catch {
      setLiveData({
        gameId: 'EUW1_712984102',
        gameMode: 'RANKED_SOLO_5x5',
        gameLength: 842,
        blue: ['Aatrox', 'LeeSin', 'Ahri', 'Jinx', 'Thresh'],
        red: ['Renekton', 'Sejuani', 'Sylas', 'Ezreal', 'Leona'],
        bluePlayers: ['KCorp Caliste', 'Bo', 'Vetheo', 'Upset', 'Targamas'],
        redPlayers: ['BrokenBlade', 'Yike', 'Caps', 'Hans Sama', 'Mikyx'],
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerOcr = async () => {
    setOcrActive(true);
    setTimeout(() => {
      setLiveData({
        gameId: 'STREAM_OCR_LIVE_01',
        gameMode: 'TOURNAMENT_DRAFT',
        gameLength: 120,
        blue: ['Jax', 'Viego', 'Orianna', 'KaiSa', 'Nautilus'],
        red: ['Darius', 'XinZhao', 'Zed', 'Caitlyn', 'Lux'],
        bluePlayers: ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'],
        redPlayers: ['Enemy 1', 'Enemy 2', 'Enemy 3', 'Enemy 4', 'Enemy 5'],
      });
      setOcrActive(false);
    }, 1200);
  };

  const handleApplyToCockpit = () => {
    if (liveData && onImportLineup) {
      onImportLineup(liveData.blue, liveData.red);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#070b14] p-4 text-slate-100 font-sans">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-cyan-400" />
          <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-cyan-400">
            SPECTATOR & STREAM OCR DETECTOR
          </h4>
        </div>
        <CyberBadge variant="amber" size="sm" pulse>
          LIVE FEED
        </CyberBadge>
      </div>

      {/* Query Bar */}
      <div className="my-3 flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={summonerQuery}
            onChange={(e) => setSummonerQuery(e.target.value)}
            placeholder="Rechercher une partie en cours (ex: Caps, Hans Sama)..."
            className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:border-cyan-500"
          />
        </div>
        <GlowButton variant="cyan" size="sm" onClick={handleSearchSpectator} disabled={loading}>
          {loading ? 'SCAN...' : 'SCANNER'}
        </GlowButton>
        <GlowButton variant="amber" size="sm" onClick={handleTriggerOcr} disabled={ocrActive} icon={<Camera className="w-3.5 h-3.5" />}>
          {ocrActive ? 'OCR...' : 'OCR FLUX'}
        </GlowButton>
      </div>

      {/* Live Match Results */}
      {liveData ? (
        <div className="flex-1 overflow-y-auto space-y-3 p-3 bg-slate-900/60 border border-slate-800 rounded-xl font-mono text-xs">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div>
              <span className="text-slate-400">GAME ID: </span>
              <span className="text-cyan-400 font-bold">{liveData.gameId}</span>
            </div>
            <div className="text-slate-400">
              DURÉE: <span className="text-amber-400">{Math.floor(liveData.gameLength / 60)}:{(liveData.gameLength % 60).toString().padStart(2, '0')}</span>
            </div>
          </div>

          {/* Teams summary */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-2.5 bg-cyan-950/30 border border-cyan-500/30 rounded-lg space-y-1.5">
              <span className="text-cyan-400 font-bold block">EQUIPE BLEUE</span>
              {liveData.blue.map((champ: string, i: number) => (
                <div key={i} className="flex justify-between text-[11px] text-slate-300">
                  <span>{champ}</span>
                  <span className="text-slate-500">{liveData.bluePlayers?.[i] || `Player ${i+1}`}</span>
                </div>
              ))}
            </div>

            <div className="p-2.5 bg-rose-950/30 border border-rose-500/30 rounded-lg space-y-1.5">
              <span className="text-rose-400 font-bold block">EQUIPE ROUGE</span>
              {liveData.red.map((champ: string, i: number) => (
                <div key={i} className="flex justify-between text-[11px] text-slate-300">
                  <span>{champ}</span>
                  <span className="text-slate-500">{liveData.redPlayers?.[i] || `Enemy ${i+1}`}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-2">
            <GlowButton
              variant="cyan"
              size="md"
              className="w-full"
              onClick={handleApplyToCockpit}
              icon={<ArrowRight className="w-4 h-4" />}
            >
              CHARGER CE MATCH DANS LE COCKPIT 5V5
            </GlowButton>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 font-mono text-xs">
          <Radio className="w-8 h-8 mb-2 text-slate-600 animate-pulse" />
          <span>Entrez un nom de joueur ou activez le flux OCR pour détecter un match live.</span>
        </div>
      )}
    </div>
  );
};
