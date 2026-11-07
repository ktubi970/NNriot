import React from 'react';
import { SystemStatus } from '../../types';
import { CyberBadge } from '../atoms/CyberBadge';
import { GlowButton } from '../atoms/GlowButton';
import {
  Cpu,
  Zap,
  Globe,
  Radio,
  MessageSquareCode,
  Eye,
  Sliders,
  Database,
  Layers,
  Sparkles,
} from 'lucide-react';

interface TelemetryHeaderProps {
  systemStatus: SystemStatus | null;
  region: string;
  onSelectRegion: (region: string) => void;
  patch: string;
  onOpenDrawerTab: (tab: 'analytics' | 'spectator' | 'trainer' | 'history') => void;
  activeDrawerTab: string | null;
}

export const TelemetryHeader: React.FC<TelemetryHeaderProps> = ({
  systemStatus,
  region,
  onSelectRegion,
  patch,
  onOpenDrawerTab,
  activeDrawerTab,
}) => {
  return (
    <header className="w-full bg-[#070b14]/95 border-b border-cyan-500/20 backdrop-blur-xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-40">
      {/* Brand & Identity */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/30 border border-cyan-500/50 shadow-[0_0_15px_rgba(0,240,255,0.4)]">
          <Layers className="w-5 h-5 text-cyan-400" />
          <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-emerald-400 rounded-full animate-ping" />
        </div>

        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-black font-mono tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500 uppercase">
              NNriot
            </h1>
            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-cyan-950/80 text-cyan-400 border border-cyan-500/40 rounded">
              v2.0 CYBER COCKPIT
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-400 block -mt-0.5">
            Deep Neural Draft Predictor & LoL Analytics Engine
          </span>
        </div>
      </div>

      {/* Telemetry Strip */}
      <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
        {/* Model status */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-900/80 border border-slate-800 rounded font-mono text-xs">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-400">MODEL:</span>
          <span className="text-emerald-400 font-bold">
            {systemStatus?.model_loaded ? 'TF RESNET READY' : 'STANDALONE ML'}
          </span>
        </div>

        {/* Vector Dimension */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 bg-slate-900/80 border border-slate-800 rounded font-mono text-xs">
          <Database className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-slate-400">VECTOR:</span>
          <span className="text-purple-400 font-bold">
            {systemStatus?.input_dimension || 512}D HASH
          </span>
        </div>

        {/* Latency / Ping */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-900/80 border border-slate-800 rounded font-mono text-xs">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-400">PING:</span>
          <span className="text-amber-400 font-bold">
            {systemStatus?.latencyMs || 12}ms
          </span>
        </div>

        {/* Region Selector */}
        <div className="flex items-center gap-1 bg-slate-900/80 border border-slate-800 rounded px-2 py-0.5 font-mono text-xs">
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <select
            value={region}
            onChange={(e) => onSelectRegion(e.target.value)}
            className="bg-transparent text-cyan-300 font-mono text-xs focus:outline-none cursor-pointer"
          >
            <option value="EUW" className="bg-slate-900">EUW</option>
            <option value="NA" className="bg-slate-900">NA</option>
            <option value="KR" className="bg-slate-900">KR</option>
            <option value="EUNE" className="bg-slate-900">EUNE</option>
          </select>
        </div>

        {/* Patch badge */}
        <CyberBadge variant="slate" size="sm">
          PATCH {patch}
        </CyberBadge>
      </div>

      {/* Cyber Drawer Trigger Buttons */}
      <div className="flex items-center gap-2">
        <GlowButton
          variant={activeDrawerTab === 'analytics' ? 'cyan' : 'slate'}
          size="sm"
          onClick={() => onOpenDrawerTab('analytics')}
          icon={<MessageSquareCode className="w-3.5 h-3.5" />}
        >
          ANALYTICS AI
        </GlowButton>

        <GlowButton
          variant={activeDrawerTab === 'spectator' ? 'cyan' : 'slate'}
          size="sm"
          onClick={() => onOpenDrawerTab('spectator')}
          icon={<Eye className="w-3.5 h-3.5" />}
        >
          SPECTATOR LIVE
        </GlowButton>

        <GlowButton
          variant={activeDrawerTab === 'trainer' ? 'cyan' : 'slate'}
          size="sm"
          onClick={() => onOpenDrawerTab('trainer')}
          icon={<Sliders className="w-3.5 h-3.5" />}
        >
          TRAINER
        </GlowButton>
      </div>
    </header>
  );
};
