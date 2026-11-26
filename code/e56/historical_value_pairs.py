"""Warm-start the E57 action-value head on prior complete one-action interventions."""
import torch
from torch.nn import functional as F

from advanced_network_runs import OUT, H80, TRAIN, VALID, CaptureValue, load, read, save, store_model, utility, value_input


def extract(cases):
    cases={c['case']:c for c in cases}
    pairs=[]
    for file in sorted((H80/'geometry_network_ablation/diagnosis').glob('*/*/result.json')):
        r=read(file);name=r['case'];t=r['first_flip']
        if name not in cases or t is None:continue
        traces=[read(file.parent/arm/'trace.json') for arm in ('reference','intervention')]
        if not all(trace['case']==cases[name]['path'] and trace['seed']==cases[name]['seed']
                   and trace['requested_cycles']==80 and (trace['cycles']==80 or len(trace['captures'])==2)
                   for trace in traces):continue
        prefix=lambda tr:[(l['plans'],l['pursuers'],l['targets']) for l in tr['logs'][:t]]
        if prefix(traces[0])!=prefix(traces[1]):continue
        records=[]
        for tr in traces:
            log=tr['logs'][t]; rr=[]
            for i in ('0','1'):
                plan=log['plans'][i]; root=log['search'][i]['root']
                hits=[j for j,a in enumerate(root) if a['endpoint']==plan['endpoint'] and a['duration']==plan['duration']]
                if len(hits)!=1:break
                idx=hits[0]; rr.append(((plan['endpoint'],plan['duration'],root[idx]['pass_through']),
                    value_input(root,idx,len(tr['logs'][t-1]['captures']) if t else 0)))
            records.append(rr)
        if any(len(rr)!=2 for rr in records):continue
        changed=[i for i in (0,1) if records[0][i][0]!=records[1][i][0]]
        if len(changed)!=1:continue
        i=changed[0];a,b=(rr[i][1] for rr in records)
        if a[:11]!=b[:11]:continue
        delta=utility(r['reference'])-utility(r['without_first_changed_action'])
        pairs.append(dict(case=name,source=str(file),a=a,b=b,delta=delta))
    return pairs


def main():
    torch.set_num_threads(1);torch.manual_seed(52)
    data={k:extract(cases) for k,cases in [('train',TRAIN),('valid',VALID)]}
    save(OUT/'historical_value_pairs.json',data)
    def tensors(rows):
        rows=[r for r in rows if r['delta']!=0]
        return (torch.tensor([r['a'] for r in rows]),torch.tensor([r['b'] for r in rows]),
                torch.tensor([float(r['delta']>0) for r in rows]))
    train,valid=map(tensors,data.values())
    info={k:{'pairs':len(rows),'informative':sum(r['delta']!=0 for r in rows)} for k,rows in data.items()}
    if not len(train[2]) or not len(valid[2]):
        save(OUT/'historical_value_fit.json',info);return
    model=load(CaptureValue(),OUT/'state.pt')
    opt=torch.optim.AdamW(model.effect.parameters(),lr=.0002)
    def loss(batch):
        a,b,y=batch
        return F.binary_cross_entropy_with_logits(model(a)-model(b),y)
    with torch.no_grad():best=float(loss(valid))
    info['initial_validation_loss']=best
    for _ in range(30):
        train_loss=loss(train);opt.zero_grad();train_loss.backward();opt.step()
        with torch.no_grad():score=float(loss(valid))
        if score<best:
            best=score;store_model('historical_value',model)
    info['best_validation_loss']=best
    save(OUT/'historical_value_fit.json',info)


if __name__=='__main__':main()
