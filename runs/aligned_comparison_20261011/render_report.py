"""Render the completed pilot's JSON summaries; never load test rows or weights.

Run with Windows Python 3.14 after evaluate.py has published comparison.json.
Only the report and its PNG/SVG figures are generated when this script runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
REPORT = ROOT / 'docs/model_serving/aligned_14d_18d_pilot_20261011.md'
FIGURE = RUN / 'pilot_comparison'
LABELS = {'14d_raw': '14D brut', '14d_calibrated': '14D calibré',
          '18d_raw': '18D brut', '18d_calibrated': '18D calibré'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid_constant)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def number(value, digits=5, *, signed=False):
    if value is None:
        return '—'
    require(math.isfinite(float(value)), 'Non-finite report value')
    return format(float(value), ('+' if signed else '') + f'.{digits}f').replace('.', ',')


def integer(value):
    return f'{int(value):,}'.replace(',', ' ')


def percent(value, digits=2):
    return '—' if value is None else number(100 * float(value), digits) + ' %'


def link(path, label):
    target = Path(os.path.relpath(Path(path), REPORT.parent)).as_posix()
    return f'[{label}]({target})'


def date_utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).strftime('%d/%m/%Y %H:%M UTC')


def passed_count(result):
    require(result.get('exit_code') == 0, 'A required test run did not pass')
    if type(result.get('passed')) is int:
        return result['passed']
    match = re.search(r'\b(\d+) passed\b', result.get('summary', ''))
    require(match is not None, 'Test JSON must include passed count or pytest summary')
    return int(match.group(1))


def conclusion(delta):
    value = float(delta['estimate'])
    low, high = map(float, delta['ci95'])
    require(low <= high and all(map(math.isfinite, (value, low, high))), 'Invalid primary interval')
    if value < 0:
        observation = 'Le 18D obtient une log-loss calibrée plus faible que le 14D sur ce test.'
    elif value > 0:
        observation = 'Le 18D obtient une log-loss calibrée plus élevée que le 14D sur ce test.'
    else:
        observation = 'Les deux modèles ont la même log-loss calibrée sur ce test.'
    if low <= 0 <= high:
        uncertainty = "L'intervalle exploratoire inclut zéro : ce pilote ne permet pas de départager clairement les deux modèles."
    elif high < 0:
        uncertainty = "L'intervalle exploratoire est entièrement négatif et va dans le sens du 18D, sans constituer une preuve confirmatoire."
    else:
        uncertainty = "L'intervalle exploratoire est entièrement positif et va dans le sens du 14D, sans constituer une preuve confirmatoire."
    return (observation + ' **Δ log-loss (18D − 14D) = ' + number(value, 6, signed=True)
            + ', IC exploratoire à 95 % [' + number(low, 6, signed=True) + ' ; '
            + number(high, 6, signed=True) + '].** ' + uncertainty)


def render_figure(comparison, trainers, baseline, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {14: '#2878A5', 18: '#B85830'}
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), layout='constrained')
        for dim, metrics in trainers.items():
            history = metrics['history']
            epochs = [row['epoch'] for row in history]
            losses = [row['val_log_loss'] for row in history]
            best = metrics['best_epoch']
            best_loss = next(row['val_log_loss'] for row in history if row['epoch'] == best)
            axes[0].plot(epochs, losses, marker='o', color=colors[dim], label=f'{dim}D')
            axes[0].scatter([best], [best_loss], marker='*', s=220, color=colors[dim],
                            edgecolors='white', linewidths=.8, zorder=4,
                            label=f'{dim}D retenu : époque {best}')
        axes[0].set(title='Validation : choix du checkpoint', xlabel='Époque',
                    ylabel='Log-loss brute — plus bas = meilleur', xticks=range(1, protocol['epochs'] + 1))
        axes[0].grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        scores = comparison['metrics']
        values = [baseline['log_loss'], scores['14d_calibrated']['model']['log_loss'],
                  scores['18d_calibrated']['model']['log_loss']]
        bars = axes[1].bar(['Fréquence du train', '14D calibré', '18D calibré'], values,
                           color=['#8B939C', colors[14], colors[18]], width=.64)
        for bar, value in zip(bars, values):
            axes[1].annotate(number(value), (bar.get_x() + bar.get_width()/2, value),
                             xytext=(0, 6), textcoords='offset points', ha='center', fontsize=10)
        axes[1].set(title=f"Test commun : {integer(comparison['test_count'])} matchs",
                    ylabel='Log-loss — plus bas = meilleur', ylim=(0, max(values) * 1.16))
        axes[1].grid(axis='y', alpha=.2)
        axes[1].set_axisbelow(True)
        fig.suptitle('Pilote aligné 14D / 18D · une seed · aucune promotion', fontsize=13)
        for extension in ('png', 'svg'):
            fig.savefig(FIGURE.with_suffix('.' + extension), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-tests-json', type=Path, default=RUN / 'training_tests.json')
    parser.add_argument('--training-tests-log', type=Path, default=None)
    args = parser.parse_args()
    comparison = read_json(RUN / 'comparison.json')
    require(comparison.get('status') == 'completed', 'Evaluation must be complete before rendering')
    require(comparison.get('production_promotion') is False, 'Unexpected production promotion')
    prep = read_json(RUN / 'preparation_result.json')
    status = read_json(RUN / 'training_status.json')
    protocol = read_json(RUN / 'protocol.json')
    trainers = {dim: read_json(RUN / f'{dim}d/model.metrics.json') for dim in (14, 18)}
    invariant_tests = read_json(RUN / 'invariant_tests.json')
    invariant_passes = passed_count(invariant_tests)
    training_passes, training_evidence = None, None
    if args.training_tests_json.is_file():
        training_passes = passed_count(read_json(args.training_tests_json))
        training_evidence = args.training_tests_json
    elif args.training_tests_log is not None:
        evidence_text = args.training_tests_log.read_text(encoding='utf-8')
        summaries = re.findall(r'(?m)^.*\b(\d+) passed\b[^\r\n]*$', evidence_text)
        require(summaries and not re.search(r'\b[1-9]\d* (?:failed|errors?)\b', evidence_text),
                'Training test log lacks an unambiguous successful summary')
        training_passes, training_evidence = int(summaries[-1]), args.training_tests_log
    require(prep.get('status') == 'verified' and status.get('status') == 'completed', 'Incomplete prerequisites')
    require(all(prep.get(key) is True for key in ('hashes_verified', 'temporal_purges_verified', 'previous_tests_excluded')),
            'Preparation verification flags missing')
    require(protocol.get('collection_restarted') is False and protocol.get('production_promotion') is False,
            'Unexpected collection/promotion policy')
    require(set(status['completed']) == {'14', '18'}, 'Both training jobs are required')
    require(comparison['invariants']['status'] == 'verified', 'Post-training invariants failed')
    require(comparison['dropped_test_rows'] == 0, 'Unexpected omitted test rows')
    require(prep['cohort']['splits']['test']['count'] == comparison['test_count'], 'Test count mismatch')
    for dim, metrics in trainers.items():
        require(metrics['player_feature_dim'] == dim and metrics['calibrated'] is True, 'Invalid trained candidate')
        require(metrics['checkpoint_sha256'] == comparison['candidates'][str(dim)]['checkpoint_sha256']
                == status['completed'][str(dim)]['checkpoint_sha256'], 'Candidate identity mismatch')
        require(status['completed'][str(dim)]['metrics_sha256'] == digest(RUN / f'{dim}d/model.metrics.json'),
                'Training metrics changed')
        require(metrics['dataset_manifest_sha256'] == prep['dataset_manifest_sha256'], 'Dataset mismatch')
        require([row['epoch'] for row in metrics['history']] == list(range(1, protocol['epochs'] + 1)),
                'Training budget differs from protocol')
    baseline = comparison['metrics']['14d_calibrated']['baseline']
    require(all(scores['baseline'] == baseline for scores in comparison['metrics'].values()), 'Baselines differ')
    initial_match = re.search(r'initial prospective protocol required (\d+)', protocol['test_size_exception'])
    require(initial_match is not None, 'Initial prospective test size is missing from protocol')
    initial_count = int(initial_match.group(1))
    delta = comparison['primary']
    cohort = prep['cohort']
    splits = cohort['splits']
    bootstrap = comparison['bootstrap_context']
    validation_end = datetime.fromisoformat(splits['validation']['last_utc'])
    test_begin = datetime.fromisoformat(splits['test']['first_utc'])
    gap_days = (test_begin - validation_end).total_seconds() / 86400
    lines = ['# Pilote aligné 14D / 18D — 11 octobre 2026', '', conclusion(delta), '',
        'La collecte est restée arrêtée. Les deux nouveaux candidats restent des artefacts de recherche ; '
        'aucun remplacement du modèle servi n’a été effectué.', '',
        'Le test de ce pilote est désormais évalué et ses matchs doivent rester protégés pour les expériences futures. '
        'Les mentions « reserved » dans les manifests immuables décrivent leur état avant cette évaluation ; '
        'le statut ultérieur est consigné dans `test_usage.json` et `comparison.json`.', '',
        '**Scores sur le même test.** Log-loss et Brier : plus bas est meilleur ; AUC et exactitude : plus haut est meilleur. '
        'La baseline est une probabilité constante estimée sur les labels du train uniquement, '
        f"P(équipe B) = {percent(comparison['baseline']['probability_team_b'])}.", '',
        '| Candidat | Log-loss | Brier | AUC | Exactitude |',
        '|---|---:|---:|---:|---:|']
    for label, score in [('Fréquence du train', baseline)] + [
            (LABELS[key], comparison['metrics'][key]['model']) for key in LABELS]:
        lines.append(f"| {label} | {number(score['log_loss'])} | {number(score['brier'])} | "
                     f"{number(score['roc_auc'], 4)} | {percent(score['accuracy'])} |")
    lines += ['', f"![Validation et comparaison test]({Path(os.path.relpath(FIGURE.with_suffix('.png'), REPORT.parent)).as_posix()})", '',
              link(FIGURE.with_suffix('.svg'), 'Graphique vectoriel SVG') + '.', '',
              '**Cohorte et protocole.** Les deux dimensions utilisent les mêmes matchs et labels, après exclusions communes.', '',
              '| Partition | Matchs | Première création | Dernière création |', '|---|---:|---|---|']
    for name, label in [('train', 'Train'), ('validation', 'Validation'), ('test', 'Test')]:
        part = splits[name]
        lines.append(f"| {label} | {integer(part['count'])} | {date_utc(part['first_utc'])} | {date_utc(part['last_utc'])} |")
    lines += ['', f"Le test utilise {integer(comparison['test_count'])} matchs prospectifs disponibles, contre "
        f"{integer(initial_count)} initialement prévus : dérogation déclarée avant entraînement. "
        f"La capture comporte {integer(cohort['source_rows'])} sources, dont {integer(protocol['warmup_rows'])} "
        f"d’amorçage ; l’historique retient les {integer(protocol['history_window_limit'])} créations antérieures "
        'et uniquement les résultats terminés avant chaque cible. '
        f"{integer(cohort['temporal_purge_count'])} cibles ont été purgées aux frontières temporelles. "
        f"{integer(cohort['exclusions'].get('timeline_unavailable_in_audit', 0))} cibles sans historique timeline "
        'ont été exclues des deux variantes. '
        f"Les {integer(protocol['prior_test_ids_protected'])} anciens IDs de test sont protégés ; "
        f"{integer(cohort['exclusions'].get('previous_test', 0))} étaient présents parmi les cibles de cette source.", '',
        f"Même builder actuel, macro de dimension 15, seed {protocol['seed']}, {protocol['epochs']} époques, "
        f"batch {protocol['batch_size']}, learning rate {number(protocol['learning_rate'], 4)}, PCGrad ; "
        'arrêt anticipé désactivé. Le meilleur checkpoint et la température sont choisis sur la validation uniquement. '
        'Les checkpoints et calibrations ont été verrouillés avant les prédictions test ; aucun réglage n’a suivi leur lecture.', '',
        '| Candidat | Époque retenue | Log-loss brute validation | Température |', '|---|---:|---:|---:|']
    for dim, metrics in trainers.items():
        lines.append(f"| {dim}D | {metrics['best_epoch']} | {number(metrics['best_val_log_loss'])} | "
                     f"{number(comparison['candidates'][str(dim)]['temperature'], 4)} |")
    lines += ['', '**Couverture sélective.** Tous les seuils fixés avant évaluation sont présentés, sans sélection du meilleur seuil. '
        'Les taux portent sur les probabilités calibrées ; les intervalles de Wilson sont descriptifs.', '',
        '| Modèle | Conviction minimale | Retenus | Couverture | Exactitude retenue | Wilson 95 % |',
        '|---|---:|---:|---:|---:|---|']
    for dim in (14, 18):
        rows = comparison['metrics'][f'{dim}d_calibrated']['selective_fixed_thresholds']
        require([row['threshold'] for row in rows] == comparison['selective_thresholds'], 'Selective thresholds differ')
        for row in rows:
            interval = ('—' if not row['count'] else
                        f"[{percent(row['wilson_lower'])} ; {percent(row['wilson_upper'])}]")
            lines.append(f"| {dim}D | {percent(row['threshold'], 0)} | {integer(row['count'])} | "
                         f"{percent(row['coverage'])} | {percent(row['accuracy'])} | {interval} |")
    invariant_panel = comparison['invariants']
    max_error = max(float(value) for model in invariant_panel['models'].values()
                    for value in model['max_absolute_errors'].values())
    test_statement = (f"{integer(invariant_passes)} tests de partitions/parité/invariants/v4 passent, "
                      'avec résultat JSON conservé. ')
    if training_passes is not None:
        test_statement += (f"Séparément, {integer(training_passes)} tests ciblés training/PCGrad passent "
                           f"({integer(invariant_passes + training_passes)} passes au total sur ces périmètres). ")
    else:
        test_statement += ('Le périmètre training/PCGrad a également donné 13 passes dans la sortie '
                           'd’exécution de cette conversation (session 17243, chunk 0a8abd, exit 0) ; '
                           'aucun log local indépendant n’a été conservé pour ce passage. ')
    artifact_links = [link(RUN / 'comparison.json', 'comparaison complète'),
        link(RUN / 'protocol.json', 'protocole'), link(RUN / 'preparation_result.json', 'préparation et replay'),
        link(RUN / 'training_status.json', 'état des entraînements'), link(RUN / 'lock.json', 'verrou des candidats'),
        link(RUN / 'test_usage.json', 'test évalué et IDs à protéger'),
        link(RUN / 'invariant_tests.json', 'tests invariants'),
        link(RUN / '14d/model.metrics.json', 'métriques 14D'), link(RUN / '18d/model.metrics.json', 'métriques 18D')]
    if training_evidence is not None:
        artifact_links.append(link(training_evidence, 'tests training/PCGrad'))
    lines += ['', '**Vérifications.** '
        + test_statement +
        f"Les {integer(prep['panel_tensor_comparisons'])} comparaisons de tenseurs sur "
        f"{integer(len(prep['independent_replay_panel']))} cibles constituent un replay échantillonné. "
        f"La parité des 14 colonnes communes et des macros est exhaustive sur "
        f"{integer(prep['common_columns_verified_targets'])} cibles. Les hashes, memberships et purges sont vérifiés. "
        f"Après entraînement, les deux modèles passent les invariants sur "
        f"{integer(len(invariant_panel['match_ids']))} cibles de validation : permutation, swap de draft, miroir neutre "
        'et biais de côté de la tête winner. '
        f"L’erreur absolue maximale observée est {max_error:.3g}, pour une tolérance {invariant_panel['atol']:.3g}. "
        f"Les deux candidats ont produit {integer(comparison['inference_count_each_model'])} prédictions test chacun, "
        f"avec {integer(comparison['dropped_test_rows'])} ligne exclue. La suite globale du dépôt n’a pas été relancée.", '',
        '**Limites.** Une seule seed par candidat : l’intervalle n’inclut pas l’aléa d’entraînement. '
        f"Le test commence {number(gap_days, 2)} jours après la fin de la validation ; "
        'ce décalage de dates et les évolutions de population/patch peuvent affecter le transfert. '
        f"Le bootstrap utilise {integer(bootstrap['repeats'])} rééchantillonnages appariés, "
        f"des blocs mobiles de {integer(bootstrap['chronological_match_block_size'])} matchs chronologiques "
        f"et la seed {bootstrap['seed']}. Le test ne représente que "
        f"{number(bootstrap['equivalent_nonoverlapping_block_count'], 2)} blocs non chevauchants équivalents : "
        'les intervalles restent exploratoires, et des joueurs communs ou dépendances temporelles peuvent traverser les blocs. '
        'Les résultats historiques sont supposés observables à la fin de partie, faute de dates d’ingestion. '
        'Les profils statiques de champions n’ont pas de provenance historique par patch. '
        'La validation a servi à sélectionner les checkpoints et à calibrer ; elle n’est pas un test indépendant. '
        'Les couvertures et intervalles descriptifs ne constituent pas une garantie de risque en service.', '',
        '**Artefacts.** ' + ', '.join(artifact_links) + '.', '']
    render_figure(comparison, trainers, baseline, protocol)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(REPORT)
    print(json.dumps({'report': str(REPORT), 'png': str(FIGURE.with_suffix('.png')),
                      'svg': str(FIGURE.with_suffix('.svg'))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
