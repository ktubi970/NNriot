import React from 'react';
import { FeatureImpact } from '../../types';
import { ListFilter } from 'lucide-react';

interface FeatureMatrixSHAPProps {
  features: FeatureImpact[];
}

export const FeatureMatrixSHAP: React.FC<FeatureMatrixSHAPProps> = ({ features }) => {
  return (
    <section className="surface-panel p-5 sm:p-6" aria-labelledby="contributions-heading">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="contributions-heading" className="result-heading">Contributions</h2>
        <span className="secondary-text text-xs">
          {features.length ? `${features.length} éléments fournis` : 'Aucune donnée fournie'}
        </span>
      </div>

      {features.length ? (
        <ul className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mt-5">
          {features.map((feature) => (
            <li key={feature.id} className="rounded-xl border border-[#3b4049] bg-[#22252b] p-4">
              <div className="flex items-start justify-between gap-3">
                <h3 className="text-sm font-medium">{feature.name}</h3>
                <span className={`text-xs font-medium tabular-nums shrink-0 ${feature.impact > 0 ? 'team-blue' : feature.impact < 0 ? 'team-red' : 'secondary-text'}`}>
                  {feature.impact > 0 ? '+' : ''}{feature.impact.toLocaleString('fr-FR', { maximumFractionDigits: 4 })}
                </span>
              </div>
              <p className="secondary-text text-xs leading-relaxed mt-2">{feature.description}</p>
              <p className="secondary-text text-[11px] mt-3">
                {feature.impact > 0 ? 'En faveur de l’équipe bleue' : feature.impact < 0 ? 'En faveur de l’équipe rouge' : 'Contribution neutre'}
              </p>
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex items-start gap-3 mt-5 py-3">
          <ListFilter className="w-5 h-5 secondary-text shrink-0 mt-0.5" aria-hidden="true" />
          <div>
            <p className="text-sm">Aucune contribution disponible</p>
            <p className="secondary-text text-xs leading-relaxed mt-1 max-w-2xl">
              Le service ne fournit pas de contributions interprétables pour ce résultat.
            </p>
          </div>
        </div>
      )}
    </section>
  );
};
