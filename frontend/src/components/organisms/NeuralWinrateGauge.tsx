import React from 'react';
import { PredictionResult } from '../../types';
import { CyberBadge } from '../atoms/CyberBadge';
import { Cpu, Zap, Activity, Clock, Crosshair, ShieldCheck } from 'lucide-react';

interface NeuralWinrateGaugeProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export const NeuralWinrateGauge: React.FC<NeuralWinrateGaugeProps> = ({
  prediction,
  isLoading,
}) => {
  const bluePercent = prediction ? Math.round(prediction.blueWinProb * 1000) / 10 : 50.0;
  const redPercent = prediction ? Math.round(prediction.redWinProb * 1000) / 10 : 50.0;
  const rawConf = prediction?.confidence ?? 0.88;
  const confidence = Math.min(99, Math.max(55, Math.round(rawConf > 1 ? (rawConf > 100 ? rawConf / 10 : rawConf) : rawConf * 100)));

  return (
    <div className="relative flex flex-col items-center justify-between p-5 bg-[#080d1a]/95 border border-cyan-500/40 rounded-xl shadow-[0_0_40px_rgba(0,240,255,0.12)] backdrop-blur-xl h-full cyber-grid overflow-hidden">
      {/* Background Neon Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top Status Strip */}
      <div className="w-full flex items-center justify-between border-b border-slate-800/80 pb-2.5 z-10">
        <div className="flex items-center gap-2">
          <Activity className={`w-4 h-4 text-cyan-400 ${isLoading ? 'animate-spin' : ''}`} />
          <span className="text-xs font-mono font-bold tracking-wider text-slate-300 uppercase">
            NEURAL ENGINE INFERENCE
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <CyberBadge variant={prediction?.modelType === 'neural_tf' ? 'green' : 'cyan'} size="sm">
            <Cpu className="w-3 h-3" />
            {prediction?.modelType === 'neural_tf' ? 'TF-2.x RESNET' : 'FAST-HASH ML'}
          </CyberBadge>
          <CyberBadge variant="slate" size="sm">
            <Zap className="w-2.5 h-2.5 text-amber-400" />
            {prediction?.latencyMs || 15}ms
          </CyberBadge>
        </div>
      </div>

      {/* Central Visual Gauge */}
      <div className="relative flex flex-col items-center justify-center my-4 z-10">
        {/* SVG Radial Gauge */}
        <div className="relative w-48 h-48 flex items-center justify-center">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle
              cx="50"
              cy="50"
              r="40"
              className="stroke-slate-800/80"
              strokeWidth="8"
              fill="transparent"
            />
            {/* Red Arc (Remaining) */}
            <circle
              cx="50"
              cy="50"
              r="40"
              className="stroke-rose-500/80"
              strokeWidth="8"
              strokeDasharray="251.2"
              strokeDashoffset="0"
              fill="transparent"
            />
            {/* Blue Arc (Dynamic) */}
            <circle
              cx="50"
              cy="50"
              r="40"
              className="stroke-cyan-400 transition-all duration-700 ease-out"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray="251.2"
              strokeDashoffset={251.2 - (251.2 * (bluePercent / 100))}
              fill="transparent"
              style={{
                filter: 'drop-shadow(0 0 8px rgba(0, 240, 255, 0.7))',
              }}
            />
          </svg>

          {/* Central Values Display */}
          <div className="absolute flex flex-col items-center justify-center text-center">
            <span className="text-[10px] font-mono tracking-widest text-slate-400 uppercase">
              VICTOIRE BLUE
            </span>
            <span className="text-3xl font-black font-mono tracking-tight text-cyan-400 drop-shadow-[0_0_12px_rgba(0,240,255,0.6)]">
              {bluePercent.toFixed(1)}%
            </span>
            <div className="flex items-center gap-1 mt-0.5">
              <span className="text-[10px] font-mono text-slate-500">RED:</span>
              <span className="text-xs font-mono font-bold text-rose-400">
                {redPercent.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Winrate Bar Meter */}
        <div className="w-full max-w-xs mt-2">
          <div className="flex justify-between text-[11px] font-mono mb-1">
            <span className="text-cyan-400 font-bold">BLUE TEAM</span>
            <span className="text-rose-400 font-bold">RED TEAM</span>
          </div>
          <div className="w-full h-2.5 bg-slate-800/90 rounded-full overflow-hidden flex border border-slate-700">
            <div
              className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 transition-all duration-500 shadow-[0_0_8px_rgba(0,240,255,0.8)]"
              style={{ width: `${bluePercent}%` }}
            />
            <div
              className="h-full bg-gradient-to-r from-rose-400 to-rose-600 transition-all duration-500 shadow-[0_0_8px_rgba(244,63,94,0.8)]"
              style={{ width: `${redPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Bottom Metrics Grid */}
      <div className="w-full grid grid-cols-3 gap-2 pt-3 border-t border-slate-800/80 z-10 font-mono">
        <div className="bg-slate-900/70 border border-slate-800 rounded p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-[10px] text-slate-400">
            <ShieldCheck className="w-3 h-3 text-emerald-400" />
            CONFIANCE
          </div>
          <span className="text-xs font-bold text-emerald-300 mt-0.5 block">
            {confidence}%
          </span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-[10px] text-slate-400">
            <Crosshair className="w-3 h-3 text-amber-400" />
            KILLS ESTIMÉS
          </div>
          <span className="text-xs font-bold text-slate-200 mt-0.5 block">
            <span className="text-cyan-400">{prediction?.blueExpectedKills ?? 18}</span>
            <span className="text-slate-500 mx-1">-</span>
            <span className="text-rose-400">{prediction?.redExpectedKills ?? 15}</span>
          </span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded p-2 text-center">
          <div className="flex items-center justify-center gap-1 text-[10px] text-slate-400">
            <Clock className="w-3 h-3 text-cyan-400" />
            DURÉE
          </div>
          <span className="text-xs font-bold text-slate-200 mt-0.5 block">
            {prediction?.expectedDuration ?? '31:45'}
          </span>
        </div>
      </div>
    </div>
  );
};
