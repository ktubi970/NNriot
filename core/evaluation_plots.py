"""Standalone Plotly risk/coverage and reliability figures for evaluation runs."""
from __future__ import annotations
import csv
import numpy as np


def write_evaluation_plots(out, cohort, predictions, metrics):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ids=metrics.get('intersection_match_ids',[])
    if not ids:return
    labels={r['match_id']:r['winner_label'] for r in cohort}
    y=np.array([labels[mid] for mid in ids]);rows=[]
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Fixed conviction thresholds','Ten equal-width reliability bins'))
    for name,score in metrics['metrics'].items():
        curve=[r for r in score['risk_coverage'] if r['risk'] is not None]
        fig.add_trace(go.Scatter(x=[r['coverage_initial_cohort'] for r in curve],y=[r['risk'] for r in curve],mode='lines+markers',name=name,legendgroup=name,customdata=[r['threshold'] for r in curve],hovertemplate='Coverage=%{x:.3f}<br>Error=%{y:.3f}<br>Conviction threshold=%{customdata}<extra>%{fullData.name}</extra>'),row=1,col=1)
        rows.extend({'model':name,**r} for r in score['risk_coverage'])
    with (out/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','threshold','count','coverage','coverage_intersection','risk','risk_wilson95','coverage_initial_cohort']);w.writeheader();w.writerows(rows)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='ideal',line={'color':'black','dash':'dash'},showlegend=False),row=1,col=2)
    for name in metrics['metrics']:
        p=np.array([predictions[name][mid] for mid in ids]);bins=np.minimum((p*10).astype(int),9)
        points=[(float(p[bins==b].mean()),float(y[bins==b].mean())) for b in range(10) if (bins==b).any()]
        fig.add_trace(go.Scatter(x=[x for x,_ in points],y=[v for _,v in points],mode='lines+markers',name=name,legendgroup=name,showlegend=False),row=1,col=2)
    fig.update_xaxes(title_text='Coverage on initial test cohort',range=[0,1],row=1,col=1)
    fig.update_yaxes(title_text='Error on retained predictions',row=1,col=1)
    fig.update_xaxes(title_text='Predicted Red probability',range=[0,1],row=1,col=2)
    fig.update_yaxes(title_text='Observed Red win frequency',range=[0,1],row=1,col=2)
    fig.update_layout(title='Exploratory common cohort — no validated model selection',template='plotly_white',height=750,legend={'orientation':'h','y':-0.25},margin={'b':180})
    fig.write_html(out/'curves.html',include_plotlyjs=True,full_html=True)
