import React from 'react';
import { Eye, Camera } from 'lucide-react';

interface LiveSpectatorTabProps {
  onImportLineup?: (blueChampions: string[], redChampions: string[]) => void;
}

export const LiveSpectatorTab: React.FC<LiveSpectatorTabProps> = () => (
  <div className="h-full space-y-5 overflow-y-auto p-4 text-slate-100">
    <div>
      <div className="flex items-center gap-2">
        <Eye aria-hidden="true" className="h-4 w-4 text-slate-400" />
        <h3 className="font-semibold">Partie en direct</h3>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-slate-400">L’import automatique d’une partie en direct n’est pas disponible dans cette interface. Vous pouvez préparer les deux équipes manuellement dans la draft.</p>
    </div>
    <section className="rounded-xl border border-slate-700/70 bg-slate-900/30 p-4">
      <h4 className="font-medium">Import de composition</h4>
      <p id="spectator-unavailable" role="status" className="mt-2 text-sm leading-relaxed text-slate-400">Aucune composition en direct n’est chargée.</p>
      <button type="button" disabled aria-describedby="spectator-unavailable" className="tool-button mt-4 min-h-11 w-full disabled:cursor-not-allowed disabled:opacity-50">Importer une partie — indisponible</button>
    </section>
    <section className="rounded-xl border border-slate-700/70 bg-slate-900/30 p-4">
      <h4 className="flex items-center gap-2 font-medium"><Camera aria-hidden="true" className="h-4 w-4 text-slate-400" />Reconnaissance d’un flux vidéo</h4>
      <p id="ocr-unavailable" className="mt-2 text-sm leading-relaxed text-slate-400">La détection des champions par OCR n’est pas connectée à une source vérifiée.</p>
      <button type="button" disabled aria-describedby="ocr-unavailable" className="tool-button mt-4 min-h-11 w-full disabled:cursor-not-allowed disabled:opacity-50">OCR du flux — indisponible</button>
    </section>
  </div>
);
