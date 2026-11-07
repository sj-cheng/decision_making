import unittest
import numpy as np
from e45g_training_acceleration import ExactArrayCache, parallel_collect


class ImmediatePool:
    def map(self, function, values):
        return map(function, values)


class TrainingAccelerationTests(unittest.TestCase):
    def test_cache_uses_exact_array_content_and_returns_mutation_isolated_values(self):
        calls=[]
        def build(value):
            calls.append(value.copy())
            return (value*2, (value+1, 7))
        cache=ExactArrayCache(build,capacity=2)
        x=np.array([1.,2.]);first=cache(x);first[0][0]=99
        first[1][0][0]=88
        second=cache(x)
        np.testing.assert_array_equal(second[0],[2.,4.])
        np.testing.assert_array_equal(second[1][0],[2.,3.])
        cache(np.nextafter(x,np.inf));cache(x.astype(np.float32))
        self.assertEqual(len(calls),3)
        self.assertEqual(len(cache.entries),2)

    def test_episode_splitting_preserves_order_with_short_successes_and_invalid_attempt(self):
        calls=[]
        durations=[10,0,10,80,80,80]
        def request(job,rng,entries):
            return dict(index=len(calls),token=int(rng.integers(2**31)))
        # Each stream owns its own counter; the request builder's counter must
        # advance at construction, independently of execution completion.
        counter=[0]
        def build(job,rng,entries):
            req=dict(index=counter[0],token=int(rng.integers(2**31)))
            counter[0]+=1
            return req
        def episode(job):
            req=job['requests'][0];calls.append(req)
            n=durations[req['index']]
            return dict(arrays={'feature':np.full((n,1),req['token'],dtype=np.float32)},
                        lengths=[n] if n else [],summaries=[req] if n else [],invalid=[] if n else [req])
        rows=parallel_collect(ImmediatePool(),[dict(seed=17,steps=256)],[],build,episode)
        expected_rng=np.random.default_rng(17)
        self.assertEqual([r['token'] for r in calls],[int(expected_rng.integers(2**31)) for _ in durations])
        self.assertEqual(rows[0]['lengths'],[10,10,80,80,80])
        self.assertEqual(len(calls),6)
        self.assertEqual([r['index'] for r in rows[0]['summaries']],[0,2,3,4,5])
        self.assertEqual([r['index'] for r in rows[0]['invalid']],[1])
        self.assertEqual(len(rows[0]['arrays']['feature']),260)


if __name__=='__main__':unittest.main()
