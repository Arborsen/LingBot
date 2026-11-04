"""CPU checks using a synthetic task map; no official data or checkpoint needed."""
import collections
import json
from pathlib import Path
import tempfile
import unittest

from stage_data import LOW_TASKS,TaskSampler,learning_rate


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'mapping.json'
        tasks=sorted(LOW_TASKS+[f'other_task_{i}' for i in range(40)])
        self.path.write_text(json.dumps(dict(tasks={t:dict(intervals=[[i*100,(i+1)*100]])for i,t in enumerate(tasks)})))

    def tearDown(self):self.tmp.cleanup()

    def sampler(self,rank=0,world=2):
        return TaskSampler(range(5000),world,rank,mapping=self.path)

    def test_weighted_distribution_and_frame_membership(self):
        sampler=self.sampler();counts=collections.Counter()
        for i in range(100000):
            task,(frame,seed)=sampler.draw(i);counts[task]+=1
            self.assertTrue(any(start<=frame<end for start,end in sampler.mapping['tasks'][task]['intervals']))
        for task,count in counts.items():
            p=(1.5 if task in LOW_TASKS else 1)/55
            self.assertLess(abs(count-100000*p),6*(100000*p*(1-p))**.5)

    def test_rank_shards_match_global_sequence(self):
        a,b,whole=self.sampler(0),self.sampler(1),self.sampler(0,1)
        for i in range(100):
            self.assertEqual(a.draw(i),whole.draw(2*i));self.assertEqual(b.draw(i),whole.draw(2*i+1))

    def test_restart_reproduces_remaining_stream(self):
        a=self.sampler();stream=iter(a)
        for _ in range(29):next(stream)
        state=a.state_dict();expected=[next(stream) for _ in range(100)]
        b=self.sampler();b.load_state_dict(state);actual=iter(b)
        self.assertEqual(expected,[next(actual) for _ in range(100)])
        changed=self.sampler();changed.fingerprint='different'
        with self.assertRaisesRegex(ValueError,'mapping_sha256'):changed.load_state_dict(state)

    def test_schedule_boundaries(self):
        for step,expected in [(15000,1.25e-5),(15200,7.5e-5),(25000,2.5e-5)]:
            self.assertAlmostEqual(learning_rate(step),expected)
        self.assertGreater(learning_rate(15100),learning_rate(15000))
        self.assertGreater(learning_rate(18000),learning_rate(24000))


if __name__=='__main__':unittest.main(verbosity=2)
