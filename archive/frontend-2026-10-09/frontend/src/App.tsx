import React, { useState, useEffect } from 'react';
import { Champion, SystemStatus } from './types';
import { fetchAllChampions, DEFAULT_CHAMPIONS } from './services/championData';
import { getSystemStatus } from './services/api';
import { usePrediction } from './hooks/usePrediction';
import { TelemetryHeader } from './components/organisms/TelemetryHeader';
import { TeamLineup } from './components/organisms/TeamLineup';
import { NeuralWinrateGauge } from './components/organisms/NeuralWinrateGauge';
import { FeatureMatrixSHAP } from './components/organisms/FeatureMatrixSHAP';
import { CyberDrawer } from './components/organisms/CyberDrawer';
import { ArrowLeftRight, Sparkles, Terminal, Activity } from 'lucide-react';
import { GlowButton } from './components/atoms/GlowButton';

export function App() {
  const [champions, setChampions] = useState<Champion[]>(DEFAULT_CHAMPIONS);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [region, setRegion] = useState('EUW');
  const [patch, setPatch] = useState('14.24');
  const [activeDrawerTab, setActiveDrawerTab] = useState<'analytics' | 'spectator' | 'trainer' | 'history' | null>(null);

  // Load champions and system status on mount
  useEffect(() => {
    fetchAllChampions().then(setChampions);
    getSystemStatus().then(setSystemStatus);

    const interval = window.setInterval(() => {
      getSystemStatus().then(setSystemStatus);
    }, 15000);

    return () => window.clearInterval(interval);
  }, []);

  const {
    blueSlots,
    redSlots,
    prediction,
    isLoading,
    updateSlot,
    randomizeTeam,
    resetTeam,
    swapTeams,
    importLineup,
  } = usePrediction(champions);

  return (
    <div className="min-h-screen flex flex-col bg-[#050811] text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">
      {/* 1. Header Telemetry */}
      <TelemetryHeader
        systemStatus={systemStatus}
        region={region}
        onSelectRegion={setRegion}
        patch={patch}
        onOpenDrawerTab={(tab) => setActiveDrawerTab(tab)}
        activeDrawerTab={activeDrawerTab}
      />

      {/* 2. Panoramic Cockpit */}
      <main className="flex-1 p-3 sm:p-4 max-w-[1780px] w-full mx-auto flex flex-col gap-4">
        {/* Upper Grid: Blue Team | Neural Gauge & Actions | Red Team */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Blue Team (5 Cols) */}
          <div className="lg:col-span-4 h-[560px]">
            <TeamLineup
              team="blue"
              title="EQUIPE BLEUE (SIDE A)"
              slots={blueSlots}
              champions={champions}
              onUpdateSlot={(id, updates) => updateSlot('blue', id, updates)}
              onRandomizeTeam={() => randomizeTeam('blue')}
              onResetTeam={() => resetTeam('blue')}
            />
          </div>

          {/* Center Neural Engine (4 Cols) */}
          <div className="lg:col-span-4 flex flex-col gap-3 h-[560px]">
            <div className="flex-1">
              <NeuralWinrateGauge prediction={prediction} isLoading={isLoading} />
            </div>

            {/* Quick Swap Controls */}
            <div className="flex items-center justify-between gap-2 p-2.5 bg-[#080d1a]/90 border border-slate-800 rounded-xl">
              <span className="text-[11px] font-mono text-slate-400 pl-1">
                PERMUTATION DES SIDES
              </span>
              <GlowButton
                variant="cyan"
                size="sm"
                onClick={swapTeams}
                icon={<ArrowLeftRight className="w-3.5 h-3.5" />}
              >
                SWAP BLUE / RED
              </GlowButton>
            </div>
          </div>

          {/* Red Team (4 Cols) */}
          <div className="lg:col-span-4 h-[560px]">
            <TeamLineup
              team="red"
              title="EQUIPE ROUGE (SIDE B)"
              slots={redSlots}
              champions={champions}
              onUpdateSlot={(id, updates) => updateSlot('red', id, updates)}
              onRandomizeTeam={() => randomizeTeam('red')}
              onResetTeam={() => resetTeam('red')}
            />
          </div>
        </div>

        {/* Lower Row: Neural SHAP Feature Explainability Matrix */}
        <div className="grid grid-cols-1 gap-4">
          <div className="h-[280px]">
            <FeatureMatrixSHAP features={prediction?.features || []} />
          </div>
        </div>
      </main>

      {/* 3. Off-Canvas Cyber Drawer (Secondary Tools) */}
      <CyberDrawer
        isOpen={Boolean(activeDrawerTab)}
        activeTab={activeDrawerTab}
        onClose={() => setActiveDrawerTab(null)}
        onSelectTab={(tab) => setActiveDrawerTab(tab)}
        onImportLineup={(blueChamps, redChamps) => {
          importLineup(blueChamps, redChamps);
          setActiveDrawerTab(null);
        }}
      />
    </div>
  );
}
export default App;
