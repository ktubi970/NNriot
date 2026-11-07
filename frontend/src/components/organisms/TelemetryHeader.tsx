import React from 'react';
import { SystemStatus } from '../../types';
import { GlowButton } from '../atoms/GlowButton';
import { Layers, ChartNoAxesCombined, Eye, Sliders, History } from 'lucide-react';

interface TelemetryHeaderProps {
  systemStatus: SystemStatus | null;
  patch: string;
  onOpenDrawerTab: (tab: 'analytics' | 'spectator' | 'trainer' | 'history') => void;
  activeDrawerTab: string | null;
}

export const TelemetryHeader: React.FC<TelemetryHeaderProps> = ({
  systemStatus,
  patch,
  onOpenDrawerTab,
  activeDrawerTab,
}) => {
  const serviceLabel = !systemStatus
    ? 'Statut du service en attente'
    : systemStatus.apiConnected === false
      ? 'Service indisponible'
      : systemStatus.model_loaded
        ? 'Modèle chargé'
        : 'Modèle indisponible';

  const tabs = [
    { id: 'analytics', label: 'Analyse', icon: ChartNoAxesCombined },
    { id: 'spectator', label: 'Partie en direct', icon: Eye },
    { id: 'trainer', label: 'Modèle', icon: Sliders },
    { id: 'history', label: 'Historique', icon: History },
  ] as const;

  return (
    <header className="cockpit-header">
      <div className="header-content">
        <div className="flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl border border-[#464c57] bg-[#2b2e35] flex items-center justify-center" aria-hidden="true">
            <Layers className="w-5 h-5 text-slate-200" />
          </span>
          <div>
            <span className="text-lg font-semibold tracking-tight">NNriot</span>
            <p className="secondary-text text-[11px]">Draft &amp; analyse</p>
          </div>
        </div>

        <nav className="header-tools" aria-label="Outils de la draft">
          {tabs.map(({ id, label, icon: Icon }) => (
            <GlowButton
              key={id}
              variant="slate"
              size="sm"
              onClick={() => onOpenDrawerTab(id)}
              icon={<Icon className="w-4 h-4" />}
              aria-haspopup="dialog"
              aria-expanded={activeDrawerTab === id}
            >
              {label}
            </GlowButton>
          ))}
        </nav>

        <details className="service-details">
          <summary>{serviceLabel}</summary>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3 pb-2 pt-1 text-xs secondary-text">
            <span>Catalogue {patch}</span>
            {systemStatus?.model_id && <span>Modèle : {systemStatus.model_id}</span>}
            {systemStatus?.feature_contract && <span>Features : {systemStatus.feature_contract}</span>}
            {systemStatus?.input_dimension != null && <span>Dimension d’entrée : {systemStatus.input_dimension}</span>}
            {systemStatus?.latencyMs != null && <span>Temps de réponse : {systemStatus.latencyMs} ms</span>}
          </div>
        </details>
      </div>
    </header>
  );
};
