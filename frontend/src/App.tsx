import { useState, useEffect } from 'react';
import { Champion, SystemStatus } from './types';
import { fetchAllChampions, DEFAULT_CHAMPIONS, DDRAGON_VERSION } from './services/championData';
import { getSystemStatus } from './services/api';
import { usePrediction } from './hooks/usePrediction';
import { TelemetryHeader } from './components/organisms/TelemetryHeader';
import { TeamLineup } from './components/organisms/TeamLineup';
import { NeuralWinrateGauge } from './components/organisms/NeuralWinrateGauge';
import { FeatureMatrixSHAP } from './components/organisms/FeatureMatrixSHAP';
import { CyberDrawer } from './components/organisms/CyberDrawer';
import { ArrowLeftRight } from 'lucide-react';
import { GlowButton } from './components/atoms/GlowButton';

export function App() {
  const [champions, setChampions] = useState<Champion[]>(DEFAULT_CHAMPIONS);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [activeDrawerTab, setActiveDrawerTab] = useState<'analytics' | 'spectator' | 'trainer' | 'history' | null>(null);

  useEffect(() => {
    let mounted = true;
    fetchAllChampions().then((catalogue) => {
      if (mounted) setChampions(catalogue);
    });
    const refreshStatus = () => {
      getSystemStatus()
        .then((status) => { if (mounted) setSystemStatus(status); })
        .catch(() => { if (mounted) setSystemStatus(null); });
    };
    refreshStatus();
    const interval = window.setInterval(refreshStatus, 15000);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const {
    blueSlots,
    redSlots,
    prediction,
    isLoading,
    error,
    updateSlot,
    randomizeTeam,
    resetTeam,
    swapTeams,
    importLineup,
  } = usePrediction(champions);

  return (
    <div className="min-h-screen">
      <a className="skip-link" href="#draft-analysis">Aller à l’analyse</a>
      <TelemetryHeader
        systemStatus={systemStatus}
        patch={DDRAGON_VERSION}
        onOpenDrawerTab={setActiveDrawerTab}
        activeDrawerTab={activeDrawerTab}
      />

      <main id="draft-analysis" className="cockpit-main" tabIndex={-1}>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="secondary-text text-xs font-medium mb-2">League of Legends · Comparaison 5 contre 5</p>
            <h1 className="page-heading">Analyse de draft</h1>
            <p className="secondary-text text-sm mt-3 max-w-xl">
              Composez les deux équipes pour comparer leurs probabilités de victoire.
            </p>
          </div>
          <span className="secondary-text text-xs">Analyse automatique après chaque modification</span>
        </div>

        <div className="draft-grid">
          <div className="draft-team-blue">
            <TeamLineup
              team="blue"
              title="Équipe bleue"
              slots={blueSlots}
              champions={champions}
              onUpdateSlot={(id, updates) => updateSlot('blue', id, updates)}
              onRandomizeTeam={() => randomizeTeam('blue')}
              onResetTeam={() => resetTeam('blue')}
            />
          </div>

          <div className="draft-team-red">
            <TeamLineup
              team="red"
              title="Équipe rouge"
              slots={redSlots}
              champions={champions}
              onUpdateSlot={(id, updates) => updateSlot('red', id, updates)}
              onRandomizeTeam={() => randomizeTeam('red')}
              onResetTeam={() => resetTeam('red')}
            />
          </div>

          <div className="draft-result">
            <NeuralWinrateGauge prediction={prediction} isLoading={isLoading} error={error} />
            <GlowButton
              variant="slate"
              onClick={swapTeams}
              icon={<ArrowLeftRight className="w-4 h-4" />}
              className="w-full"
            >
              Inverser les équipes
            </GlowButton>
          </div>
        </div>

        <div className="mt-6">
          <FeatureMatrixSHAP features={prediction?.features || []} />
        </div>
      </main>

      <CyberDrawer
        isOpen={Boolean(activeDrawerTab)}
        activeTab={activeDrawerTab}
        onClose={() => setActiveDrawerTab(null)}
        onSelectTab={setActiveDrawerTab}
        onImportLineup={(blueChamps, redChamps) => {
          importLineup(blueChamps, redChamps);
          setActiveDrawerTab(null);
        }}
      />
    </div>
  );
}
export default App;
