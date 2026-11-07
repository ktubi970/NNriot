import React, { useState, useRef, useId } from 'react';
import { AnalyticsChatMessage } from '../../types';
import { queryAnalyticsChat } from '../../services/api';
import { GlowButton } from '../atoms/GlowButton';
import { MessageSquare, Send } from 'lucide-react';

interface ChatMessage extends Omit<AnalyticsChatMessage, 'evidence'> {
  evidence?: Record<string, unknown>[];
  warnings?: string[];
  abstained?: boolean;
  datasetVersion?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

const EvidenceCard: React.FC<{ packet: Record<string, unknown> }> = ({ packet }) => {
  const sample = isRecord(packet.sample) ? packet.sample : {};
  const estimate = isRecord(packet.estimate) ? packet.estimate : {};
  const interval = estimate.confidence_interval_95;
  const limitations = Array.isArray(packet.limitations) ? packet.limitations.filter((value): value is string => typeof value === 'string') : [];
  return (
    <div className="space-y-1 border-t border-slate-700/70 pt-3 text-xs text-slate-400">
      {typeof packet.evidence_id === 'string' && <p className="break-all">Référence : {packet.evidence_id}</p>}
      {finiteNumber(sample.matches) && <p>Échantillon : {sample.matches.toLocaleString()} matchs</p>}
      {finiteNumber(estimate.value) && <p>Estimation : {estimate.value} {typeof estimate.unit === 'string' ? estimate.unit : ''}</p>}
      {finiteNumber(estimate.q_value) && <p>q-value : {estimate.q_value}</p>}
      {Array.isArray(interval) && finiteNumber(interval[0]) && finiteNumber(interval[1]) && <p>Intervalle à 95 % : [{interval[0]}, {interval[1]}]</p>}
      {limitations.length > 0 && <p>Limites : {limitations.join(' · ')}</p>}
    </div>
  );
};

export const AnalyticsChatTab: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pendingRef = useRef(false);
  const inputId = useId();

  const sampleQuestions = [
    'Quel est l’impact du premier Dragon ?',
    'Comment comparer deux compositions ?',
    'Quelles synergies sont présentes dans cette draft ?',
  ];

  const handleSend = async (questionText?: string) => {
    const query = (questionText ?? input).trim();
    if (!query || pendingRef.current) return;
    pendingRef.current = true;
    setMessages(previous => [...previous, {
      id: `user_${Date.now()}`, sender: 'user', text: query, timestamp: new Date().toLocaleTimeString(),
    }]);
    if (!questionText) setInput('');
    setLoading(true);
    setError(null);
    try {
      const response = await queryAnalyticsChat(query);
      const answer = response.answer || response.response;
      if (typeof answer !== 'string' || !answer.trim()) throw new Error('Le service analytique n’a pas retourné de réponse.');
      setMessages(previous => [...previous, {
        id: `assistant_${Date.now()}`, sender: 'assistant', text: answer,
        timestamp: new Date().toLocaleTimeString(),
        evidence: Array.isArray(response.evidence) ? response.evidence.filter(isRecord) : undefined,
        warnings: Array.isArray(response.warnings) ? response.warnings.filter(isRecord).map(warning => typeof warning.message === 'string' ? warning.message : typeof warning.code === 'string' ? warning.code : '').filter(Boolean) : undefined,
        abstained: response.abstained === true,
        datasetVersion: typeof response.dataset_version === 'string' ? response.dataset_version : undefined,
      }]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Le service analytique est indisponible. Réessayez plus tard.');
    } finally {
      pendingRef.current = false;
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col p-4 text-slate-100">
      <div className="flex items-center gap-2">
        <MessageSquare aria-hidden="true" className="h-4 w-4 text-slate-400" />
        <h3 className="font-semibold">Analyse de la draft</h3>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-slate-400">Posez une question au service analytique. Les statistiques sont affichées uniquement lorsqu’elles sont fournies par le service.</p>
      <div className="flex shrink-0 flex-wrap gap-1.5 py-3" aria-label="Exemples de questions">
        {sampleQuestions.map(question => (
          <button key={question} type="button" disabled={loading} onClick={() => handleSend(question)} className="tool-button min-h-10 text-left text-xs disabled:opacity-50">{question}</button>
        ))}
      </div>
      <div role="log" aria-label="Conversation analytique" aria-live="polite" aria-relevant="additions" aria-busy={loading} className="my-2 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {messages.length === 0 && <p className="rounded-xl border border-slate-700/70 bg-slate-900/30 p-4 text-sm text-slate-400">La conversation apparaîtra ici.</p>}
        {messages.map(message => (
          <article key={message.id} className={`rounded-xl border p-3 text-sm leading-relaxed ${message.sender === 'user' ? 'ml-8 border-slate-600 bg-slate-700/35' : 'mr-4 border-slate-700/70 bg-slate-900/40'}`}>
            <div className="mb-2 flex items-center justify-between gap-2 text-xs text-slate-400">
              <span>{message.sender === 'user' ? 'Vous' : 'Service analytique'}</span>
              <time>{message.timestamp}</time>
            </div>
            <p className="whitespace-pre-wrap">{message.text}</p>
            {message.abstained && <p className="mt-3 text-xs text-amber-200">Le service s’abstient de conclure.</p>}
            {message.warnings?.map((warning, index) => <p key={index} className="mt-2 text-xs text-amber-200">{warning}</p>)}
            {message.evidence && message.evidence.length > 0 && <details className="mt-3"><summary className="min-h-11 cursor-pointer py-3 text-xs font-medium text-slate-300">Statistiques retournées par le service</summary><div className="space-y-3">{message.evidence.map((packet, index) => <EvidenceCard key={typeof packet.evidence_id === 'string' ? packet.evidence_id : index} packet={packet} />)}{message.datasetVersion && <p className="break-all text-xs text-slate-400">Version des données : {message.datasetVersion}</p>}</div></details>}
          </article>
        ))}
      </div>
      {loading && <p role="status" className="py-2 text-sm text-slate-400">Consultation du service analytique…</p>}
      {error && <p role="alert" className="mb-3 rounded-lg border border-amber-700/50 bg-amber-950/20 p-3 text-sm text-amber-200">{error}</p>}
      <form onSubmit={event => { event.preventDefault(); void handleSend(); }} className="border-t border-slate-700/70 pt-3">
        <label htmlFor={inputId} className="mb-2 block text-sm text-slate-300">Votre question</label>
        <div className="flex items-center gap-2">
          <input id={inputId} type="text" value={input} onChange={event => setInput(event.target.value)} placeholder="Poser une question…" className="field-input min-w-0 flex-1" />
          <GlowButton type="submit" variant="slate" size="sm" disabled={loading || !input.trim()} icon={<Send aria-hidden="true" className="h-4 w-4" />}>Envoyer</GlowButton>
        </div>
      </form>
    </div>
  );
};
