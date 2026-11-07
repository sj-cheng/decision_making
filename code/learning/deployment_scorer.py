"""Small, nested scene-grouped feasibility screen; no actor or PPO updates."""
import hashlib
import numpy as np

METHODS=('compact_ridge','observation_ridge','compact_stump')
THRESHOLDS=(0.,.05,.1,.2,float('inf'))
SEED=50020261024


def folds(identities,count=5,salt='outer'):
    if len(set(identities))!=len(identities):
        raise ValueError('Dataset must have exactly one row per physical scene')
    if len(identities)<count:
        raise ValueError('Not enough physical scenes')
    order=sorted(range(len(identities)),key=lambda i:hashlib.sha256(
        f'{SEED}|{salt}|{identities[i]}'.encode()).digest())
    assignment=np.empty(len(order),dtype=int)
    for rank,index in enumerate(order):assignment[index]=rank%count
    return assignment


def fit(method,x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite scorer training data')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1.
    z=(x-mean)/scale
    if method.endswith('ridge'):
        alpha=100. if method=='observation_ridge' else 10.
        intercept=float(y.mean());centered=y-intercept
        dual=np.linalg.solve(z@z.T+alpha*np.eye(len(z)),centered)
        return dict(method=method,mean=mean,scale=scale,coef=z.T@dual,intercept=intercept)
    if method!='compact_stump':raise ValueError('Unknown scorer')
    best=None;error=float(np.sum((y-y.mean())**2))
    # A single split, fixed minimum support; all thresholds use training only.
    for j in range(z.shape[1]):
        for threshold in np.unique(np.quantile(z[:,j],[.2,.35,.5,.65,.8])):
            left=z[:,j]<=threshold
            if min(left.sum(),(~left).sum())<12:continue
            a,b=float(y[left].mean()),float(y[~left].mean())
            loss=float(np.sum((y-np.where(left,a,b))**2))
            if loss<error-1e-12:
                best=(j,float(threshold),a,b);error=loss
    return dict(method=method,mean=mean,scale=scale,split=best,intercept=float(y.mean()))


def predict(model,x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite scorer input')
    z=(x-model['mean'])/model['scale']
    if model['method'].endswith('ridge'):
        return np.clip(z@model['coef']+model['intercept'],-1.,1.)
    if model['split'] is None:return np.full(len(x),model['intercept'])
    j,threshold,a,b=model['split']
    return np.where(z[:,j]<=threshold,a,b)


def select_on_training(features,y,identities):
    assignment=folds(identities,4,'inner')
    choices=[]
    for method in METHODS:
        x=features['observation' if method=='observation_ridge' else 'compact']
        predictions=np.zeros(len(y))
        for fold in range(4):
            train=assignment!=fold;test=~train
            model=fit(method,x[train],y[train]);predictions[test]=predict(model,x[test])
        for threshold in THRESHOLDS:
            gain=float(np.mean(np.where(predictions>threshold,y,0.)))
            # Conservative tie handling prefers no replacement, then simpler model.
            choices.append((gain,float(threshold),-METHODS.index(method),method))
    _,threshold,_,method=max(choices)
    return method,threshold


def paired_interval(values,draws=20000,seed=SEED):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed);boot=np.empty(draws)
    for start in range(0,draws,1000):
        indices=rng.integers(0,len(values),(min(1000,draws-start),len(values)))
        boot[start:start+len(indices)]=values[indices].mean(axis=1)
    return np.quantile(boot,[.025,.975]).tolist()


def gate(result):
    return (result['delta_vs_A0']>=.03 and result['ci95_vs_A0'][0]>0 and
            result['delta_vs_AQ']>0 and result['ci95_vs_AQ'][0]>0 and
            sum(x>0 for x in result['outer_fold_delta'])>=3 and result['illegal_delta']<=.02)


def evaluate(dataset):
    identities=[row['physical_hash'] for row in dataset]
    compact=np.asarray([row['compact'] for row in dataset],float)
    observation=np.asarray([row['observation'] for row in dataset],float)
    y=np.asarray([np.mean(row['paired_delta']) for row in dataset],float)
    delta=np.asarray([row['paired_delta'] for row in dataset],float)
    illegal=np.asarray([row['illegal_delta'] for row in dataset],float)
    if delta.shape!=(len(dataset),3) or illegal.shape!=delta.shape or not np.isfinite(delta).all():
        raise ValueError('Missing candidate/stream pairing')
    assignment=folds(identities);choice=np.zeros(len(dataset),bool);predictions=np.zeros(len(dataset))
    selected=[]
    for fold in range(5):
        train=assignment!=fold;test=~train
        features=dict(compact=compact[train],observation=observation[train])
        method,threshold=select_on_training(features,y[train],[identities[i] for i in np.flatnonzero(train)])
        x=observation if method=='observation_ridge' else compact
        model=fit(method,x[train],y[train]);predictions[test]=predict(model,x[test])
        choice[test]=predictions[test]>threshold
        selected.append(dict(fold=fold,method=method,threshold=None if np.isinf(threshold) else threshold,
            train_scenes=int(train.sum()),test_scenes=int(test.sum()),
            test_replacements=int(choice[test].sum())))
    applied=delta*choice[:,None];per_scene=applied.mean(axis=1)
    versus_aq=(applied-delta).mean(axis=1)
    result=dict(physical_scenes=len(dataset),streams=3,bootstrap_draws=20000,bootstrap_seed=SEED,
        delta_vs_A0=float(per_scene.mean()),ci95_vs_A0=paired_interval(per_scene),
        delta_vs_AQ=float(versus_aq.mean()),ci95_vs_AQ=paired_interval(versus_aq),
        constant_AQ_delta=float(delta.mean()),illegal_delta=float((illegal*choice[:,None]).mean()),
        gained=int((applied>0).sum()),lost=int((applied<0).sum()),replacements=int(choice.sum()),
        outer_fold_delta=[float(per_scene[assignment==i].mean()) for i in range(5)],
        selected=selected,ordinary_intervals_only=True,independent_confirmation=False,
        scope='Exploratory nested physical-scene cross-validation; no future labels in deployment features',
        predictions=[dict(physical_hash=identity,fold=int(assignment[i]),prediction=float(predictions[i]),
            choose_AQ=bool(choice[i]),paired_delta=delta[i].tolist()) for i,identity in enumerate(identities)])
    result['passed']=gate(result)
    result['reason']='feasible_for_preregistered_new_L_data' if result['passed'] else 'no_reliable_group_out_selection_gain'
    return result
