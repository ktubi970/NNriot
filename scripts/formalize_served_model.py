"""Regenerate auditable current-runtime contracts and candidate manifests.

Does not modify checkpoints, databases, calibration or serving configuration.
Regeneration does not establish historical training feature provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.candidate_inventory import inventory_candidates, serving_manifest


def feature_spec():
    # Ordered continuous features mirror PlayerPriorsTracker, not an inferred
    # checkpoint format. Formulas and explicit cold defaults are part of v1.
    priors = [
        ('smooth_wr','probability','(wins+10*0.5)/(games+10)',.5),
        ('smooth_cwr','probability','(champion_wins+5*0.5)/(champion_games+5)',.5),
        ('champ_exp','log games','log1p(champion_games)',0),
        ('rec_wr','probability','(recent_wins+5*0.5)/(recent_games+5); last 20 results in end order',.5),
        ('role_fam','fraction','(role_games+1)/(games+5)',.2),
        ('tot_exp','log games','log1p(games)',0),
        ('norm_kda','ratio','min(2,((kills+assists+5)/(deaths+2))/5)',.5),
        ('norm_cspm','ratio','min(1.5,((cs+60)/(minutes+10))/10)',.6),
        ('norm_gpm','ratio','min(1.5,((gold+3500)/(minutes+10))/500)',.7),
        ('early_aggression','events/minute','min(1.5,(kills+deaths+1)/(minutes+2))',.35),
    ]
    timeline = [
        ('norm_gd15','2000 gold units','clip((sum_gd15/(timeline_count+5))/2000,-1,1)',0),
        ('norm_csd15','30 CS units','clip((sum_csd15/(timeline_count+5))/30,-1,1)',0),
        ('norm_xpd15','1500 XP units','clip((sum_xpd15/(timeline_count+5))/1500,-1,1)',0),
        ('norm_fbi10','fraction','clip((sum_fbi10+5*0.1)/(timeline_count+5),0,1)',.1),
    ]
    prefix=[{'name':'champion_id','unit':'registry integer ID','normalization':'none; embedded by model','missing':0},
            {'name':'role_id','unit':'integer 0..4','normalization':'none; embedded by model','missing':'error'}]
    suffix=[{'name':'lane_matchup_delta','unit':'probability difference','normalization':'neutral legacy lane tracker, exactly 0','missing':0},
            {'name':'champ_base_wr','unit':'probability','normalization':'static hand-authored profile; historical patch provenance unknown','missing':.5}]
    values=lambda xs:[{'name':n,'unit':u,'normalization':f,'missing':d} for n,u,f,d in xs]
    common={'dtype':'float32','team_shape':['batch',5,'D'],'role_order':['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY'],
            'team_perspective':{'team_a':'Blue, Riot teamId=100','team_b':'Red, Riot teamId=200'},
            'missing_policy':'Cold defaults only for unknown player history; wrong lengths/duplicate roles/nonfinite data are errors',
            'availability':'Only historical results with valid explicit result_end_ms strictly before prediction_time_ms; same latest 10000 global CLASSIC creation-row window for service and evaluation; target labels do not govern historical result inclusion',
            'cutoff':'Historical match game_creation_time; live explicit prediction_time_ms or gameCreation, otherwise request time UTC ms',
            'duration_normalization':'Historical duration seconds / 60 clipped to 5..60 minutes; missing duration explicit fallback 25 minutes',
            'macro_clash':{'shape':['batch',15],'dtype':'float32','perspective':'team_b minus team_a',
                           'order':['ad_ratio','ap_ratio','true_ratio','cc_score','scaling_score','ranged_ratio','is_full_ad','poke_siege','hard_engage','pick_catch','split_push','hyper_scaling','spike_15m','spike_25m','spike_35m'],
                           'unit':'dimensionless domain trait/archetype scores; no extra normalization'}}
    contracts={}
    for dim,extra in ((14,[]),(18,timeline)):
        contracts[f'nnriot.players{dim}.v1']={**common,'D':dim,'ordered_features':prefix+values(priors)+values(extra)+suffix}
    contracts['nnriot.players18.v1']['additional_availability']='Requires nonempty valid completed historical timeline metrics; declared per-player defaults, combat cold-start count and usable timeline-match count; unavailable timeline source rejected'
    contracts['legacy.tree24.v1']={
        'use':'exploratory evaluation only; never selected implicitly for neural serving',
        'shape':['batch',24], 'dtype':'categorical strings converted to float64 by the checkpoint-bound OrdinalEncoder',
        'ordered_features':[f'{team}_{i}_{field}' for team in ('team_a','team_b') for i in range(5) for field in ('champion','role')]+['game_mode','game_version','season_id','region'],
        'role_order':common['role_order'], 'team_perspective':common['team_perspective'],
        'normalization':'Exact legacy extract_tabular_features and its saved encoder; no padding, fitting or replacement encoder',
        'context_sources':{'game_mode':'Riot gameMode', 'game_version':'exact Riot gameVersion string',
                          'season_id':'integer major gameVersion component as a string',
                          'region':'explicit region or canonical Riot match-ID platform prefix via core/collection/regions.py'},
        'missing_policy':'Missing context becomes literal UNKNOWN; unseen categories encode to saved unknown_value=-1. Unknown platform never defaults to EUW.',
        'outputs':{'winner':'predict_proba classes [0,1] = [Blue,Red]; raw only, no calibration'},
    }
    return {'schema_version':1,'contracts':contracts,'spatial46':{'status':'unsupported','reason':'No checkpoint-bound spatial encoder training bounds or embedding provenance; never substitute zero vectors'},
            'outputs':{'winner':{'shape':['batch',2],'classes':['team_a_Blue','team_b_Red'],'positive_class':1,'calibration':'winner only; binary temperature sigmoid(logit(p_B)/T); checkpoint/split/fit verified'},
                       'draft_advantage':{'shape':['batch',2],'classes':['team_a','team_b'],'calibration':'none; raw pure composition, anti-symmetric, no DirectionalSideBias; null for legacy v3'},
                       'auxiliary_heads':'heads_raw exposes checkpoint output values without calibration or fabricated defaults; class order/dimensions in output_heads.md; no validated actionability claim',
                       'decision':'conviction abs(p_B-0.5), thresholds 0.04/0.08 existing defaults, force abstention without verified calibration AND independent serving validation'}}


def run(out):
    out.mkdir(parents=True,exist_ok=True);(out/'manifests').mkdir(exist_ok=True)
    def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    items=inventory_candidates();save(out/'candidate_inventory.json',items)
    save(out/'feature_contracts.v1.json',feature_spec())
    for item in items:
        if item.get('feature_contract') in ('nnriot.players14.v1','nnriot.players18.v1'):
            manifest=serving_manifest(item);save(out/'manifests'/(item['model_id']+'.json'),manifest)
            if item['model_id']=='candidate_deepsets_v3':save(out/'reference.manifest.json',manifest)
    print(f'Wrote {len(items)} candidate inventory entries and feature contracts to {out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='docs/model_serving')
    args=parser.parse_args();run(Path(args.output_dir))
