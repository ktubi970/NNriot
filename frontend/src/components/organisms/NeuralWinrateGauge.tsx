import React from 'react';
import { PredictionResult } from '../../types';
import { ArrowLeftRight, LoaderCircle, TriangleAlert, CircleCheck, CircleMinus } from 'lucide-react';

interface NeuralWinrateGaugeProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
  error?: string | null;
}

const percent = (value: number) => `${(value * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} %`;

export const NeuralWinrateGauge: React.FC<NeuralWinrateGaugeProps> = ({
  prediction,
  isLoading,
  error = null,
}) => {
  const hasResult = Boolean(prediction && !isLoading && !error);
  const calibrated = prediction?.calibratedProbabilities != null;
  const verified = calibrated && prediction?.calibrationStatus === 'artifact_verified';
  const decision = prediction?.selectiveDecision;

  return (
    <section
      className="surface-panel result-panel"
      aria-labelledby="prediction-result-heading"
      aria-busy={isLoading}
      data-testid="prediction-result"
    >
      <div className="flex items-center justify-between gap-3">
        <h2 id="prediction-result-heading" className="result-heading">Résultat</h2>
        <span className="secondary-text text-xs">{hasResult ? 'Probabilité de victoire' : 'Comparaison de draft'}</span>
      </div>

      {isLoading ? (
        <div className="result-empty" role="status">
          <LoaderCircle className="w-7 h-7 secondary-text animate-spin motion-reduce:animate-none" aria-hidden="true" />
          <p className="font-medium">Analyse en cours</p>
          <p className="secondary-text text-sm max-w-xs">Le modèle compare les deux compositions.</p>
        </div>
      ) : error ? (
        <div className="result-empty" role="alert">
          <TriangleAlert className="w-7 h-7 secondary-text" aria-hidden="true" />
          <p className="font-medium">Analyse indisponible</p>
          <p className="secondary-text text-sm max-w-xs">{error}</p>
        </div>
      ) : prediction ? (
        <div className="mt-6" aria-live="polite" aria-atomic="true">
          <p className="secondary-text text-xs font-medium">
            {calibrated ? 'Probabilités calibrées' : 'Probabilités brutes'}
          </p>
          <div className="grid grid-cols-2 gap-3 mt-4 mb-5">
            <div>
              <p className="team-blue text-xs mb-2">Équipe bleue</p>
              <output className="probability-value team-blue" aria-label="Probabilité de victoire de l’équipe bleue">
                {percent(prediction.blueWinProb)}
              </output>
            </div>
            <div className="text-right">
              <p className="team-red text-xs mb-2">Équipe rouge</p>
              <output className="probability-value team-red" aria-label="Probabilité de victoire de l’équipe rouge">
                {percent(prediction.redWinProb)}
              </output>
            </div>
          </div>

          <div className="flex h-2 overflow-hidden rounded-full bg-[#353a44]" aria-hidden="true">
            <div className="bg-[#80b5ff]" style={{ width: `${prediction.blueWinProb * 100}%` }} />
            <div className="bg-[#f4a1a7]" style={{ width: `${prediction.redWinProb * 100}%` }} />
          </div>

          <p className="secondary-text text-xs mt-3">
            {verified ? 'Calibration validée' : calibrated ? 'Calibration non validée' : 'Calibration indisponible'}
          </p>

          <div className="decision-note mt-6">
            <div className="flex items-center gap-2 font-medium text-sm">
              {decision?.abstained
                ? <CircleMinus className="w-4 h-4 shrink-0" aria-hidden="true" />
                : <CircleCheck className="w-4 h-4 shrink-0" aria-hidden="true" />}
              <p>{decision?.abstained ? 'Abstention du modèle' : decision?.isActionable ? 'Signal exploitable' : 'Décision indisponible'}</p>
            </div>
            <p className="secondary-text text-xs leading-relaxed mt-2">
              {decision?.recommendation || 'Aucune recommandation disponible.'}
            </p>
            {!calibrated && (
              <p className="secondary-text text-xs leading-relaxed mt-2">
                Les probabilités brutes ne suffisent pas à valider un signal exploitable.
              </p>
            )}
          </div>

          <dl className="grid grid-cols-2 gap-4 mt-5 text-xs">
            <div>
              <dt className="secondary-text">Conviction</dt>
              <dd className="text-sm font-medium mt-1">{decision?.conviction != null ? percent(decision.conviction) : 'Non fournie'}</dd>
            </div>
            <div className="text-right">
              <dt className="secondary-text">Seuil appliqué</dt>
              <dd className="text-sm font-medium mt-1">{decision?.thresholdApplied != null ? percent(decision.thresholdApplied) : 'Non fourni'}</dd>
            </div>
          </dl>

          <details className="result-details">
            <summary>Détails du résultat</summary>
            <dl className="mt-2">
              <div className="metric-row"><dt>Probabilité brute · bleue</dt><dd>{percent(prediction.rawProbabilities.blue)}</dd></div>
              <div className="metric-row"><dt>Probabilité brute · rouge</dt><dd>{percent(prediction.rawProbabilities.red)}</dd></div>
              <div className="metric-row"><dt>Statut de calibration</dt><dd>{prediction.calibrationStatus}</dd></div>
              <div className="metric-row"><dt>Tier de décision</dt><dd>{decision?.tier || 'Non fourni'}</dd></div>
              {prediction.modelId && <div className="metric-row"><dt>Modèle</dt><dd>{prediction.modelId}</dd></div>}
              <div className="metric-row"><dt>Temps de réponse</dt><dd>{prediction.latencyMs} ms</dd></div>
              <div className="metric-row"><dt>Analyse reçue à</dt><dd>{prediction.timestamp}</dd></div>
              {prediction.draftAdvantage && (
                <>
                  <div className="metric-row"><dt>Composition · bleue</dt><dd>{percent(prediction.draftAdvantage.blue)}</dd></div>
                  <div className="metric-row"><dt>Composition · rouge</dt><dd>{percent(prediction.draftAdvantage.red)}</dd></div>
                </>
              )}
            </dl>
          </details>
        </div>
      ) : (
        <div className="result-empty">
          <span className="w-12 h-12 rounded-2xl border border-[#414752] bg-[#24272e] flex items-center justify-center" aria-hidden="true">
            <ArrowLeftRight className="w-5 h-5 secondary-text" />
          </span>
          <p className="font-medium">Complétez les deux équipes</p>
          <p className="secondary-text text-sm max-w-xs">
            Choisissez cinq champions par équipe. Le résultat apparaîtra ici.
          </p>
        </div>
      )}
    </section>
  );
};
