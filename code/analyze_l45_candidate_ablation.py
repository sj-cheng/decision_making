"""Read-only scientific analysis of committed candidate-ablation suffixes."""
from pathlib import Path
import numpy as np
import run_l45_candidate_ablation as m

def main():
    args=m.parse_args(['report']);cases=m.prepare(args);items=m.requests(args,cases)
    result=m.locked_read(args.output/'statistics.json');diag=m.locked_read(args.output/'diagnostics.json')
    records=m.locked_read(args.output/'paired_results.json')
    index={(r['uid'],r['scene_id'],r['stream']):r for r in records}
    disagreements=[]
    for key,r in index.items():
        if key[0]!='G':continue
        ref=index[('P',key[1],key[2])]
        if r['success']>ref['success']:disagreements.append((key[1],key[2]))
    disagreements.sort()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'figures';out.mkdir(exist_ok=True)
    item_index={(i['uid'],i['task']['scene_id'],i['task']['stream']):i for i in items}
    illustrations=[]
    for scene,stream in disagreements:
        fig,axes=plt.subplots(1,2,figsize=(10,4.7))
        for ax,u in zip(axes,('G','P')):
            row,d=m.read_episode(args,item_index[(u,scene,stream)]);p,_=m.o.restore_scenario(d['scenario'])
            state=d['initial']['state'];active=[r for r in range(4) if p.is_active(state,r)]
            for box in p.obstacles:ax.add_patch(plt.Rectangle((box[0,0],box[1,0]),box[0,1]-box[0,0],box[1,1]-box[1,0],fc='0.75'))
            states=np.array([z['state'].reshape(-1) for z in d['trajectory']]+[d['final']['state'].reshape(-1)])
            for robot,color,name in zip(active,('#347fa3','#d47435'),('Evader','Pursuer')):
                xy=states[:,2*robot:2*robot+2];ax.plot(xy[:,0],xy[:,1],color=color,label=name,lw=1.7)
                ax.scatter(*xy[0],color=color,marker='o',s=30);ax.scatter(*xy[-1],color=color,marker='x',s=35)
            if u=='P':
                root=d['roots'][0];paths=root['paths'];audit=root['routes']
                if paths.size:
                    for route in np.unique(paths[:,0]):
                        q=paths[(paths[:,0]==route)&(paths[:,1]==0)];ax.plot(q[:,2],q[:,3],'k--',alpha=.35,lw=1)
                if audit.size:
                    q=audit[audit[:,8]>.5];ax.scatter(q[:,4],q[:,5],color='purple',marker='*',s=65,label='Predicted intercept')
            ax.set(xlim=p.state_lims[0],ylim=p.state_lims[1],aspect='equal',title=f'{u}: success={row["legal_double_capture"]}, cost={row["completion_cost"]-row["first_capture_time"]:.0f}')
            ax.legend(fontsize=8)
        fig.suptitle(f'Identical takeover state; stream={stream}, first capture={d["initial"]["time"]:.0f}')
        fig.tight_layout();name=f'G_won_P_lost_{scene[:10]}_s{stream}.png';fig.savefig(out/name,dpi=180);plt.close(fig)
        illustrations.append(dict(scene=scene,stream=stream,figure=str(out/name)))
    if disagreements:
        scene,stream=disagreements[0];fig,axes=plt.subplots(2,3,figsize=(12,7.4))
        for ax,u in zip(axes.flat,m.UIDS):
            _,d=m.read_episode(args,item_index[(u,scene,stream)]);p,_=m.o.restore_scenario(d['scenario']);step=d['trajectory'][0]
            state=step['state'];robot=next(r for r in p.pursuers if p.is_active(state,r));evader=next(r for r in p.evaders if p.is_active(state,r))
            at=state[p.state_idxs[robot],0];delta=state[p.state_idxs[evader],0]-at;root=d['roots'][0]
            for box in p.obstacles:ax.add_patch(plt.Rectangle((box[0,0],box[1,0]),box[0,1]-box[0,0],box[1,1]-box[1,0],fc='0.8'))
            actions=root['table'][:,-p.action_dim:][:,p.action_idxs[robot]]
            sources=np.asarray(root['sources']) if len(root['sources']) else np.zeros(len(actions))
            for color,mask in [('#aaaaaa',sources!=2),('#d89834',sources==2)]:
                aa=actions[mask];ax.quiver(np.full(len(aa),at[0]),np.full(len(aa),at[1]),aa[:,0],aa[:,1],angles='xy',scale_units='xy',scale=1,color=color,alpha=.45,width=.004)
            action=step['action'][p.action_idxs[robot],0]
            ax.quiver(*at,*action,angles='xy',scale_units='xy',scale=1,color='black',width=.012)
            goal=at+delta/max(1e-9,np.linalg.norm(delta))*1.8;ax.plot([at[0],goal[0]],[at[1],goal[1]],'b--',lw=1)
            ax.scatter(*at,color='black',s=20);ax.set(xlim=(at[0]-2,at[0]+2),ylim=(at[1]-2,at[1]+2),aspect='equal',title=u)
        fig.suptitle('First root candidates: gray original, orange added, black selected; blue target direction')
        fig.tight_layout();fig.savefig(out/'first_root_candidates.png',dpi=180);plt.close(fig)
    interpretations=[
        '满幅神经均值方向候选F已经覆盖原P的成功率收益；无需未来逃逸路线预测即可获得本批强改善。',
        'F-S条件增益23.74pp，普通95%区间[17.17,30.30]，五项主比较同时下界18.18pp；47个状态×流挽回，0丢失。',
        'F-S也把25%均匀候选替换为神经均值方向候选，尚不能把全部提升精确归为幅度。最小补充是同槽位、同均值方向、同扰动但不放大均值的M组。',
        'D比F仅多1个成功状态×流，但平均后段成本降低2.40步，95%区间[-3.32,-1.53]；直接追踪方向主要体现为更快收尾。',
        'G与D在198个状态×流上的成功/失败完全一致，G平均后段成本再降低0.80步，95%区间[-1.37,-0.31]。零成功率差区间[0,0]不证明总体等效。',
        'G无基础候选回退，D回退2853次；绕障碍方向减少不可用候选并缩短收尾，但本批没有新增成功。',
        'P相对G没有挽回，丢失3个状态×流；后段成本增加2.45步，95%区间[1.68,3.25]。当前预测截点没有额外价值证据。',
        'PR相对P净增2个状态×流，成功率区间跨零；后段成本增加1.67步，95%区间[0.38,2.95]。不支持继续扩大引导rollout。',
        'S原流76场、新流68场，两流平均72%；没有重新训练或更换模型，这一变化来自预登记搜索随机流。六组在各流保持配对。',
        '本轮属于旧100场景的机制开发；99个首捕状态是统计单位，198条后缀不是198个独立物理场景。全部方案仍是1000模拟神经MCTS，D/G不等于无搜索几何控制器。']
    m.o.write_json(args.output/'mechanism_interpretation.json',dict(interpretations=interpretations,
        representative_disagreements=illustrations,posthoc_analysis=True,extra_environment_episodes=0))
    lines=['','机制解释与下一步：','']+[f'{i+1}. {x}' for i,x in enumerate(interpretations)]
    lines+=['','完成情况：1188条科学后缀请求，297条源记录复用，891条新续跑；另24条隔离smoke、8项针对性检查和1条工程重放。',
        '完整性复核：1188条缓存/轨迹、297条复用来源及最终随机状态、21294次追捕搜索的1000模拟预算均通过；无追捕价值调用，引导前缀仅PR启用且最长8步。',
        '建议下一项单独登记M与F的匹配幅度对照：99状态×2流，仅新增198条后缀，F复用。其后再独立验证简化候选。此建议尚未执行。',
        '图：G成功而P失败的三条状态×流均有轨迹图，见mechanism_interpretation.json。',
        '', '![六组结果与配对区间](figures/mechanism_ablation.png)',
        '', '![同一首捕状态的根候选](figures/first_root_candidates.png)']
    path=args.output/'REPORT.md';content=path.read_text();marker='\n机制解释与下一步：'
    if marker in content:content=content.split(marker)[0]
    path.write_text(content+'\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('根候选消融已完成。\n\n'+'\n\n'.join(interpretations)+'\n')

if __name__=='__main__':main()
