import React from 'react';
import { FeatureImpact } from '../../types';
import { CyberBadge } from '../atoms/CyberBadge';
import { BrainCircuit, Info, TrendingUp, TrendingDown, Layers, Zap } from 'lucide-react';

interface FeatureMatrixSHAPProps {
  features: FeatureImpact[];
}

export const FeatureMatrixSHAP: React.FC<FeatureMatrixSHAPProps> = ({ features }) => {
  return (
    <div className="flex flex-col h-full bg-[#080d1a]/85 border border-slate-800 rounded-xl p-4 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-purple-400" />
          <h3 className="font-mono font-bold tracking-wider uppercase text-sm text-slate-200">
            MATRICE D'EXPLICABILITÉ IA (SHAP)
          </h3>
        </div>
        <CyberBadge variant="purple" size="sm">
          {features.length} FEATURES CLÉS
        </CyberBadge>
      </div>

      {/* Feature List */}
      <div className="flex flex-col gap-2.5 overflow-y-auto flex-1 pr-1">
        {features.map((feat) => {
          const isBlue = feat.impact >= 0;
          const impactPercent = (Math.abs(feat.impact) * 10).toFixed(1);

          return (
            <div
              key={feat.id}
              className="group bg-slate-900/60 hover:bg-slate-900/90 border border-slate-800/80 hover:border-slate-700 rounded-lg p-2.5 transition-all"
            >
              {/* Feature Title & Impact Tag */}
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-1.5 min-w-0">
                  {isBlue ? (
                    <TrendingUp className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  ) : (
                    <TrendingDown className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                  )}
                  <span className="text-xs font-semibold text-slate-200 truncate font-mono">
                    {feat.name}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  <span
                    className={`text-xs font-mono font-bold ${
                      isBlue ? 'text-cyan-400' : 'text-rose-400'
                    }`}
                  >
                    {isBlue ? `+${impactPercent}% (BLUE)` : `+${impactPercent}% (RED)`}
                  </span>
                </div>
              </div>

              {/* Progress Bar showing bias */}
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex my-1">
                {isBlue ? (
                  <div
                    className="h-full bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(0,240,255,0.6)]"
                    style={{ width: `${Math.min(Math.abs(feat.impact) * 35, 100)}%` }}
                  />
                ) : (
                  <div
                    className="h-full bg-rose-500 rounded-full ml-auto shadow-[0_0_8px_rgba(244,63,94,0.6)]"
                    style={{ width: `${Math.min(Math.abs(feat.impact) * 35, 100)}%` }}
                  />
                )}
              </div>

              {/* Description */}
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                {feat.description}
              </p>
            </div>
          );
        })}

        {features.length === 0 && (
          <div className="flex flex-col items-center justify-center h-48 text-center text-slate-500 font-mono text-xs">
            <Layers className="w-8 h-8 mb-2 text-slate-600 animate-pulse" />
            <span>Sélectionnez des champions pour générer la matrice neuronale.</span>
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="pt-3 mt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-500">
        <span className="flex items-center gap-1">
          <Info className="w-3 h-3 text-cyan-400" />
          Gradient SHAP calculé sur 512 dimensions hashées
        </span>
        <CyberBadge variant="slate" size="sm">VECTOR 512D</CyberBadge>
      </div>
    </div>
  );
};
