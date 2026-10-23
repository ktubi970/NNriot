"""HTTP fixture exercising the actual facade, not the obsolete trainer fallback."""
from types import SimpleNamespace
import numpy as np


ROLES = ['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY']
CHAMPIONS = {
    100: ['Aatrox', 'LeeSin', 'Ahri', 'Jinx', 'Leona'],
    200: ['Garen', 'Graves', 'Zed', 'Ashe', 'Lux'],
}


def full_match():
    return {'participants':[{'teamId':tid, 'puuid':f'{tid}_{i}', 'championName':CHAMPIONS[tid][i],
                             'teamPosition':role, 'kills':99999, 'goldEarned':999999}
                            for tid in (100, 200) for i,role in enumerate(ROLES)],
            'teams':[{'teamId':100},{'teamId':200}]}


def custom_match():
    return {key:[{'puuid':f'{tid}_{i}','champion_name':CHAMPIONS[tid][i],'role':role}
                 for i,role in enumerate(ROLES)]
            for key,tid in (('blue_team',100),('red_team',200))}


def install_serving_fixture(monkeypatch, raw, tmp_path, *, calibrated=True, validated=True, temperature=1.0):
    """Load real manifest/calibration evidence, replacing only neural inference."""
    import tensorflow as tf
    import core.prediction_service as serving
    from app.routes import api
    from tests.serving_artifact_fixture import build_serving_artifacts

    artifacts = build_serving_artifacts(
        tmp_path, calibrated=calibrated, validated=validated,
        temperature=temperature, output_heads=list(raw),
    )

    class Model:
        inputs = [SimpleNamespace(name=k,shape=(None,5,14)) for k in ('team_a','team_b')]
        output = {key: SimpleNamespace(shape=(None,) + value.shape[1:]) for key, value in raw.items()}
        def predict(self,x,verbose=0):
            return {k:v.copy() for k,v in raw.items()}
        def __call__(self,x,training=False):
            n=len(x['team_a'])
            return {k:np.repeat(v,n,axis=0) for k,v in raw.items()}
    with monkeypatch.context() as model_patch:
        model_patch.setattr(tf.keras.models, 'load_model', lambda *args, **kwargs: Model())
        service = serving.PredictionService(
            checkpoint_path=artifacts['checkpoint'], manifest_path=artifacts['manifest'],
        )
    monkeypatch.setattr(serving,'get_prediction_service',lambda **kwargs:service)
    monkeypatch.setattr(api,'get_prediction_service',lambda **kwargs:service)
    monkeypatch.setattr(api.database,'get_stale_puuids',lambda *args,**kwargs:[])
    return service
