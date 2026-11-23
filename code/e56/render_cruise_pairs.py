"""Synchronized E52-style GIF of the 16 paired E56 waypoint games."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw, ImageFont

ARCHIVE = Path(__file__).resolve().parents[2] / 'current/E系列实验输出'
ROOT = ARCHIVE / 'e56_network_screen'
RUN = ROOT / 'cruise_ab_150'
BASELINE = ROOT / 'd070c02b0be6ad33/1.50_none_256'
REFERENCE = ARCHIVE / 'e52_combined_20261118/shared_motion_v15_both_waypoint/e52_both_waypoint_dense_00_AB_25fps.gif'
OUT = RUN / 'E56_16组原候选与巡航候选_同步轨迹.gif'
SIDE, PAIR, ROW = 226, 496, 305
WIDTH, HEIGHT = PAIR * 4 + 16, ROW * 4 + 96
LABELS = ('P1', 'P2', 'E1', 'E2')
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT = ImageFont.truetype(CJK, 16, index=2)
HEAD = ImageFont.truetype(CJK, 24, index=2)
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
TAG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 10)


def panel(trace_path, x, y, draw):
    trace = json.loads(trace_path.read_text())
    case = json.loads(Path(trace['case']).read_text())
    def pixel(points):
        scale = np.array([SIDE, -SIDE]) / 20
        return np.rint(np.array([x + SIDE/2, y + SIDE/2]) + np.asarray(points)*scale).astype(int)
    draw.rectangle((x,y,x+SIDE,y+SIDE), fill=1, outline=4)
    for k in range(1,10):
        offset = round(k*SIDE/10)
        draw.line((x+offset,y+1,x+offset,y+SIDE-1), fill=13)
        draw.line((x+1,y+offset,x+SIDE-1,y+offset), fill=13)
    for (lx,rx),(by,ty) in case['map']:
        corners = pixel([[lx,ty],[rx,by]])
        draw.rectangle(tuple(corners.ravel().tolist()), fill=3, outline=4)
    events = {e['target']:round(e['time']*100) for e in trace['captures']}
    end = max(events.values()) if len(events)==2 else trace['cycles']*100
    tracks = []
    initial = case['worlds'][0]['initial']
    for i, j in enumerate((4,6,0,2)):
        pieces = [pixel(initial[j:j+2])[None,:]]
        for row in trace['logs']:
            if i<2:
                points = row['minco'][str(i)]['executed_positions']
            else:
                points = row['evader_minco'].get(str(i-2),{}).get('positions',[row['targets'][i-2]])
            xy = pixel(points)
            pieces.append(xy[np.minimum(np.arange(1,101),len(xy)-1)])
        tracks.append(np.concatenate(pieces)[:end+1])
    return tracks,events,end,(x,y)


def main():
    with Image.open(REFERENCE) as ref:
        palette = ref.getpalette()
    base = Image.new('P',(WIDTH,HEIGHT),0)
    base.putpalette(palette)
    draw = ImageDraw.Draw(base)
    draw.text((16,5),'E56  |  16 组配对追逃轨迹',font=HEAD,fill=2)
    draw.text((16,42),'2 对 2　·　追击限速 1.125 m/s　·　逃跑限速 1.5 m/s　·　双方加速度上限 1.5 m/s²',font=FONT,fill=14)
    for i,label in enumerate(LABELS):
        left=820+i*65
        draw.ellipse((left,47,left+10,57),fill=5+i)
        draw.text((left+16,43),label,font=SMALL,fill=2)
    draw.text((1100,40),'⊗ 已捕获　　每组左为原候选，右为新增巡航候选',font=FONT,fill=2)
    panels=[]
    rows=sorted(json.loads((RUN/'paired.json').read_text()),key=lambda r:r['case'])
    assert len(rows)==16
    for n,row in enumerate(rows):
        name=row['case']
        x=16+n%4*PAIR
        y=79+n//4*ROW
        stem,seed=name.rsplit('_',1)
        draw.text((x,y),f'{stem}  ·  seed {seed}',font=SMALL,fill=2)
        for idx,folder in enumerate((BASELINE,RUN)):
            left=x+idx*238
            draw.text((left,y+17),('原候选','巡航候选')[idx],font=FONT,fill=14)
            record=panel(folder/name/'trace.json',left,y+45,draw)
            assert record[2]==round(row[('original','cruise')[idx]]['double']*100)
            panels.append(record)
    previous=None
    temp=OUT.with_suffix('.rendering')
    with temp.open('wb') as stream:
        for tick in range(0,16001,10):
            for tracks,events,end,_ in panels:
                if 0<tick<end+10:
                    for i,track in enumerate(tracks):
                        points=track[min(tick-10,end):min(tick,end)+1]
                        if len(points)>1:
                            draw.line(list(map(tuple,points.tolist())),fill=9+i,width=2)
            im=base.copy()
            view=ImageDraw.Draw(im)
            view.text((1690,34),f'{tick/100:05.1f} s   |   2.5×',fill=2,font=HEAD)
            for tracks,events,end,(x,y) in panels:
                t=min(tick,end)
                for i,track in enumerate(tracks):
                    px,py=map(int,track[t])
                    captured=i>=2 and events.get(i-2,math.inf)<=t
                    view.ellipse((px-4,py-4,px+4,py+4),fill=1 if captured else 5+i,outline=5+i)
                    if captured:
                        for sign in (-1,1):
                            view.line((px-3,py-3*sign,px+3,py+3*sign),fill=5+i)
                    else:
                        view.text((px+5,py-8),LABELS[i],font=TAG,fill=5+i,stroke_width=1,stroke_fill=1)
                count=sum(v<=t for v in events.values())
                view.text((x,y+SIDE+3),f'{t/100:06.2f} s   |   Captured {count}/2',font=SMALL,fill=2)
            duration=1500 if tick==16000 else 40
            if previous is None:
                for chunk in GifImagePlugin.getheader(im,info=dict(loop=0))[0]:
                    stream.write(chunk)
                patch=im
                offset=(0,0)
                opts={}
            else:
                arr=np.asarray(im)
                diff=arr!=np.asarray(previous)
                ys=np.flatnonzero(diff.any(axis=1))
                xs=np.flatnonzero(diff.any(axis=0))
                rect=np.s_[ys[0]:ys[-1]+1,xs[0]:xs[-1]+1]
                patch=Image.fromarray(np.where(diff[rect],arr[rect],15).astype('uint8'))
                patch.putpalette(palette)
                opts=dict(transparency=15)
                offset=int(xs[0]),int(ys[0])
            for chunk in GifImagePlugin.getdata(patch,offset=offset,duration=duration,disposal=1,**opts):
                stream.write(chunk)
            previous=im
        stream.write(b';')
    with Image.open(temp) as result:
        assert result.n_frames==1601 and result.size==(WIDTH,HEIGHT)
        result.seek(1600)
        result.load()
    temp.replace(OUT)
    print(OUT,OUT.stat().st_size)


if __name__=='__main__':
    main()
