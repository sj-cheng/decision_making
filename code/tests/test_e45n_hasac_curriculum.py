import unittest
import numpy as np
import run_e45n_hasac_curriculum as n


class CurriculumTests(unittest.TestCase):
    def test_only_environment_entry_and_opponent_change(self):
        e=dict(id='original',path='/hidden',curriculum=dict(static='/static',moving='/moving'))
        self.assertEqual(n.phase_entry(e,n.PHASES[-1]),e)
        self.assertEqual(n.phase_entry(e,n.PHASES[0])['path'],'/static')
        self.assertEqual(n.phase_entry(e,n.PHASES[1])['path'],'/moving')
        self.assertEqual(n.PHASES[-1]['opponent'],'rootpuct')
        self.assertEqual(sum(p['steps'] for p in n.PHASES),150000)

    def test_visible_gate_cannot_silently_pass_failure(self):
        for successes in range(7):
            row=dict(deterministic=dict(double=successes,episodes=6))
            self.assertEqual(n.gate_passed(row),successes>=5)
        self.assertTrue(n.gate_passed(dict(deterministic=dict(double=0,episodes=2)),smoke=True))

    def test_selected_curriculum_starts_are_real_sensor_measurements(self):
        cfg=n.m.verify(n.m.OUT)
        for e in cfg['entries']:
            for phase in n.PHASES[:2]:
                packet=n.j.read(n.phase_entry(e,phase)['path'])
                self.assertEqual(packet['curriculum_variant'],phase['variant'])
                self.assertEqual(len(packet['history']),1)
                obs=packet['history'][0]
                for target in range(2):
                    self.assertTrue(any(x is not None for x in obs['reports'][target]))
                self.assertTrue(all(obs['active_e']));self.assertTrue(all(obs['active_p']))


if __name__=='__main__':unittest.main()
