import React, { useRef, useState } from 'react';
import { triggerManualTraining } from '../../services/api';
import { GlowButton } from '../atoms/GlowButton';
import { Sliders, Play, CheckCircle, AlertCircle } from 'lucide-react';

export const ModelTrainerTab: React.FC = () => {
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<'success' | 'error' | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const pendingRef = useRef(false);

  const handleManualStep = async () => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    setIsTraining(true);
    setOutcome(null);
    setTrainingStatus('Cycle d’entraînement en cours…');
    try {
      const response = await triggerManualTraining();
      if (response.success !== true) throw new Error(typeof response.error === 'string' ? response.error : typeof response.message === 'string' ? response.message : 'Le serveur n’a pas confirmé l’entraînement.');
      const count = typeof response.records_processed === 'number' ? ` ${response.records_processed} enregistrements traités.` : '';
      setTrainingStatus(`Cycle terminé avec succès.${count}`);
      setOutcome('success');
    } catch (cause) {
      setTrainingStatus(cause instanceof Error ? cause.message : 'Le service d’entraînement est indisponible.');
      setOutcome('error');
    } finally {
      pendingRef.current = false;
      setIsTraining(false);
    }
  };

  return (
    <div className="h-full space-y-5 overflow-y-auto p-4 text-slate-100">
      <div>
        <div className="flex items-center gap-2">
          <Sliders aria-hidden="true" className="h-4 w-4 text-slate-400" />
          <h3 className="font-semibold">Entraînement du modèle</h3>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">Un cycle manuel utilise les nouvelles données disponibles sur le serveur. Il démarre uniquement quand vous le demandez.</p>
      </div>
      <section className="rounded-xl border border-slate-700/70 bg-slate-900/30 p-4">
        <h4 className="font-medium">Métriques d’entraînement</h4>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">Les métriques détaillées et la configuration d’entraînement ne sont pas exposées dans cette vue.</p>
      </section>
      <section className="space-y-4 rounded-xl border border-slate-700/70 bg-slate-900/30 p-4">
        <div>
          <h4 className="font-medium">Cycle manuel</h4>
          <p className="mt-2 text-sm leading-relaxed text-slate-400">Le résultat affiché sera celui du service d’entraînement.</p>
        </div>
        <GlowButton variant="slate" size="md" onClick={handleManualStep} disabled={isTraining} icon={<Play aria-hidden="true" className="h-4 w-4" />}>
          {isTraining ? 'Entraînement en cours…' : 'Lancer un cycle manuel'}
        </GlowButton>
        {trainingStatus && (
          <div role={outcome === 'error' ? 'alert' : 'status'} className={`flex items-start gap-2 rounded-lg border p-3 text-sm ${outcome === 'error' ? 'border-amber-700/50 bg-amber-950/20 text-amber-200' : 'border-slate-600 bg-slate-800/40 text-slate-200'}`}>
            {outcome === 'success' && <CheckCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />}
            {outcome === 'error' && <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />}
            <p>{trainingStatus}</p>
          </div>
        )}
      </section>
    </div>
  );
};
