"""Two-cycle integration smoke using the real camera/LiDAR/MINCO simulator."""
import argparse
from run_e50_distributed_dynamic_smoke import run

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--output',required=True)
    p.add_argument('--case',default='current/E系列实验输出/e50_deployment_audit_20261117/formal_random_cases/wall/e49_validation_24_long_wall_0001_random_00_random_03.json')
    p.add_argument('--checkpoint'); a=p.parse_args()
    r=run(a.case,a.output,cycles=a.cycles,e52=True,e52_checkpoint=a.checkpoint,rootpuct=True)
    print(dict(output=a.output,cycles=r['cycles'],rootpuct_failures=r['rootpuct_failures'],captures=len(r['captures'])))
