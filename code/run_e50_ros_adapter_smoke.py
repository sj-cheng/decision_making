"""Dependency-free replay of ROS-shaped pose/scan/belief messages."""
import argparse, json
from pathlib import Path
import numpy as np
from solvers.e50_runtime_adapter import E50RuntimeAdapter

class Time:
    def __init__(self, sec, nanosec=0): self.sec, self.nanosec = sec, nanosec
class Header:
    def __init__(self, stamp, frame_id='base_link'): self.stamp, self.frame_id = stamp, frame_id
class Position:
    def __init__(self, x, y): self.x, self.y = x, y
class Pose:
    def __init__(self, x, y): self.position = Position(x, y)
class Poses:
    def __init__(self): self.header=Header(Time(1, 500000000)); self.poses=[Pose(0,0),Pose(1,1)]
class Scan:
    def __init__(self): self.header=Header(Time(1,500000000),'laser'); self.angle_min=-1.; self.angle_increment=.5; self.ranges=[2.,3.,4.]; self.range_max=10.
class Belief:
    def __init__(self): self.header=Header(Time(1,500000000)); self.particles=np.zeros((2,2,2)); self.weights=np.full((2,2),.5); self.active=np.array([True,True]); self.confidence=np.array([.8,.7])

def run(output):
    a=E50RuntimeAdapter(); a.update_pose_message(Poses()); a.update_scan_message(0,Scan()); a.update_scan_message(1,Scan()); a.update_belief_message(Belief())
    live=a.snapshot(1.55); stale=a.snapshot(2.)
    command=np.ones((2,2))
    result=dict(stage='ros_shaped_adapter_smoke', live=dict(degraded=live.degraded, reason=live.reason, gated=a.gate_command(live,command).tolist()), stale=dict(degraded=stale.degraded, reason=stale.reason, gated=a.gate_command(stale,command).tolist()), ros_dependency='none_for_replay; bind native ROS message objects at runtime')
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2)+'\n'); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args(); print(json.dumps(run(a.output),indent=2))
