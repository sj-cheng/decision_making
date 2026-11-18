"""Render the executed prefixes from distributed smoke logs."""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw


def render(log_path, case_path, output, fps=8):
    data=json.loads(Path(log_path).read_text()); case=json.loads(Path(case_path).read_text())
    boxes=case['map']; logs=data['logs']; W=720; pad=48
    coords=[]
    for row in logs:
        coords += row.get('pursuers', []) + row.get('targets', [])
    for b in boxes: coords += [b[0], b[1]]
    lo=min(min(p[0] for p in coords), -10.); hi=max(max(p[0] for p in coords), 10.)
    ly=min(min(p[1] for p in coords), -10.); hy=max(max(p[1] for p in coords), 10.)
    def xy(p): return (pad+(p[0]-lo)/(hi-lo)*(W-2*pad), W-pad-(p[1]-ly)/(hy-ly)*(W-2*pad))
    frames=[]; paths=[[],[]]
    for row in logs:
        for i,p in enumerate(row.get('pursuers', [])): paths[i].append(p)
        im=Image.new('RGB',(W,W+42),'white'); d=ImageDraw.Draw(im)
        for b in boxes:
            a=xy([b[0][0],b[1][0]]); c=xy([b[0][1],b[1][1]])
            d.rectangle((min(a[0],c[0]),min(a[1],c[1]),max(a[0],c[0]),max(a[1],c[1])),fill=(215,215,215),outline=(90,90,90))
        for path,color in zip(paths,[(35,95,190),(220,80,45)]):
            if len(path)>1: d.line([xy(p) for p in path],fill=color,width=3)
        for i,p in enumerate(row.get('pursuers', [])):
            q=xy(p); d.ellipse((q[0]-7,q[1]-7,q[0]+7,q[1]+7),fill=[(35,95,190),(220,80,45)][i])
        for i,p in enumerate(row.get('targets', [])):
            q=xy(p); d.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),fill=(30,160,70),outline=(0,80,0))
        title=f"cycle {row['cycle']}  captures {len(row.get('captures', []))}  collision {row.get('pursuer_collision', False)}"
        d.text((12,W+10),title,fill=(0,0,0)); frames.append(im)
    if not frames: raise ValueError('empty log')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=int(1000/fps),loop=0,optimize=False)
    return dict(output=str(output),frames=len(frames),cycles=len(logs))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--log',required=True); p.add_argument('--case',required=True); p.add_argument('--output',required=True); p.add_argument('--fps',type=int,default=8)
    print(render(p.parse_args().log,p.parse_args().case,p.parse_args().output,p.parse_args().fps))
