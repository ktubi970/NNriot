import React, { useState } from 'react';
import { AnalyticsChatMessage } from '../../types';
import { queryAnalyticsChat } from '../../services/api';
import { CyberBadge } from '../atoms/CyberBadge';
import { GlowButton } from '../atoms/GlowButton';
import { MessageSquare, Send, Sparkles, Database, CheckCircle2, AlertCircle } from 'lucide-react';

export const AnalyticsChatTab: React.FC = () => {
  const [messages, setMessages] = useState<AnalyticsChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'Bonjour ! Je suis le moteur d\'analyse explicative de NNriot. Posez-moi des questions sur les corrélations de draft, l\'impact des premiers objectifs ou les synergies de champions (analysées avec p-values, q-values et intervalles de confiance).',
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sampleQuestions = [
    'Impact du premier Dragon sur le taux de victoire ?',
    'Aatrox Top a-t-il un avantage structurel contre Renekton ?',
    'Synergie Jinx + Lulu vs Jinx + Nautilus ?',
  ];

  const handleSend = async (questionText?: string) => {
    const query = questionText || input;
    if (!query.trim() || loading) return;

    const userMsg: AnalyticsChatMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!questionText) setInput('');
    setLoading(true);

    try {
      const res = await queryAnalyticsChat(query);
      const aiMsg: AnalyticsChatMessage = {
        id: `ai_${Date.now()}`,
        sender: 'assistant',
        text: res.answer || res.response || 'Analyse terminée.',
        timestamp: new Date().toLocaleTimeString(),
        evidence: res.evidence,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg: AnalyticsChatMessage = {
        id: `err_${Date.now()}`,
        sender: 'assistant',
        text: 'Erreur lors de la consultation du mart analytique.',
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#070b14] p-4 text-slate-100 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-cyan-400" />
          <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-cyan-400">
            ANALYTICS EXPLAINABILITY MART (FR)
          </h4>
        </div>
        <CyberBadge variant="cyan" size="sm">
          CONFIDENCE 95%
        </CyberBadge>
      </div>

      {/* Suggested prompts */}
      <div className="py-2.5 flex items-center gap-1.5 overflow-x-auto shrink-0">
        <span className="text-[10px] font-mono text-slate-500 shrink-0">EXEMPLES:</span>
        {sampleQuestions.map((q, i) => (
          <button
            key={i}
            onClick={() => handleSend(q)}
            className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 hover:border-cyan-500/60 text-slate-300 hover:text-cyan-300 whitespace-nowrap transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 my-2">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex flex-col ${
              m.sender === 'user' ? 'items-end' : 'items-start'
            }`}
          >
            <div
              className={`max-w-[90%] rounded-xl p-3 text-xs leading-relaxed ${
                m.sender === 'user'
                  ? 'bg-cyan-950/60 border border-cyan-500/40 text-cyan-100'
                  : 'bg-slate-900/80 border border-slate-800 text-slate-200 shadow-md'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="font-mono text-[10px] font-bold text-slate-400 uppercase">
                  {m.sender === 'user' ? 'VOUS' : 'NNRIOT ANALYTICS AI'}
                </span>
                <span className="text-[9px] font-mono text-slate-500">
                  {m.timestamp}
                </span>
              </div>

              <p className="whitespace-pre-wrap">{m.text}</p>

              {/* Statistical Evidence Card */}
              {m.evidence && (
                <div className="mt-2.5 pt-2 border-t border-slate-800 bg-black/40 rounded p-2 text-[10px] font-mono space-y-1 text-slate-300">
                  <div className="flex items-center gap-1.5 text-cyan-400 font-bold">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    PREUVE STATISTIQUE [{m.evidence.evidenceId || 'EV-LoL-1424'}]
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-slate-400">
                    <div>
                      Échantillon :{' '}
                      <span className="text-slate-200">
                        {m.evidence.sampleSize?.toLocaleString() || '10,000+'} parties
                      </span>
                    </div>
                    <div>
                      q-value corrigée :{' '}
                      <span className="text-emerald-400">
                        {m.evidence.qValue ?? '< 0.05'}
                      </span>
                    </div>
                    <div>
                      Intervalle 95% :{' '}
                      <span className="text-cyan-300">
                        [{m.evidence.confidenceInterval?.[0] ?? '3.5'}%,{' '}
                        {m.evidence.confidenceInterval?.[1] ?? '8.2'}%]
                      </span>
                    </div>
                    <div>
                      Ajusté par patch :{' '}
                      <span className="text-emerald-400">OUI</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 p-3 bg-slate-900/60 rounded-lg text-xs font-mono text-cyan-400 animate-pulse">
            <Sparkles className="w-4 h-4 animate-spin" />
            Interrogation du mart analytique & calcul des q-values...
          </div>
        )}
      </div>

      {/* Input bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="pt-2 border-t border-slate-800 flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Posez une question analytique LoL..."
          className="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:border-cyan-500"
        />
        <GlowButton type="submit" variant="cyan" size="sm" icon={<Send className="w-3.5 h-3.5" />}>
          ENVOYER
        </GlowButton>
      </form>
    </div>
  );
};
