import json
from pathlib import Path
import tempfile
import unittest

from aggregate_scene_campaign import load_result,inference_seed,aggregate


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.task='dump_bin_bigbin'
        self.dir=self.root/'eval_results'/self.task;self.dir.mkdir(parents=True)
        self.rows=[dict(task=self.task,episode=i,seed=100000+i,task_config='demo_randomized',instruction_type='unseen',
            inference_seed=inference_seed('demo_randomized',self.task,100000+i),success=i%2==0,
            cumulative_successes=(i+2)//2,video_recording=False,scene_repair_version=1,initial_state_verified=True)
            for i in range(50)]
        self.result=dict(validated=True,episodes=50,successes=25,source=str(self.root))
        (self.dir/'_result.txt').write_text('Instruction Type: unseen\n0.5\n')
        self.write()

    def tearDown(self):self.temp.cleanup()

    def write(self):
        (self.dir/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in self.rows))

    def test_corrected_ledger_is_accepted(self):
        rows,audit=load_result(self.result,'clean2random',self.task,True)
        self.assertEqual(len(rows),50);self.assertTrue(audit['repaired'])

    def test_partial_run_cannot_be_merged(self):
        self.rows.pop();self.write()
        with self.assertRaisesRegex(AssertionError,'incomplete episodes'):load_result(self.result,'clean2random',self.task,True)

    def test_wrong_inference_seed_cannot_be_merged(self):
        self.rows[25]['inference_seed']+=1;self.write()
        with self.assertRaisesRegex(AssertionError,'inference seed'):load_result(self.result,'clean2random',self.task,True)

    def test_uncertified_retest_cannot_replace_history(self):
        self.rows[25]['initial_state_verified']=False;self.write()
        with self.assertRaises(AssertionError):load_result(self.result,'clean2random',self.task,True)

    def test_full_campaign_replaces_only_two_randomized_tasks_and_preserves_history(self):
        cfg=dict(baseline25000=str(self.root/'base25'),remaining22000=str(self.root/'base22'),
            repair25000=str(self.root/'repair25'),repair22000=str(self.root/'repair22'),
            repaired_tasks=['dump_bin_bigbin','put_bottles_dustbin'])
        tasks=cfg['repaired_tasks']+[f'task{i}' for i in range(48)]
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        write(self.root/'campaign.json',cfg);write(self.root/'base25/plan.json',dict(tasks=tasks))
        for step in [22000,25000]:
            for repair in [False,True]:
                folder=self.root/(('repair' if repair else 'base')+str(step//1000))
                status=dict(phase='complete' if repair or step==25000 else 'needs_scene_repair',settings={})
                for setting,scene,lang in [('clean2clean','demo_clean','seen'),('clean2random','demo_randomized','unseen')]:
                    if repair and setting=='clean2clean':continue
                    selected=tasks[:2] if repair else tasks
                    if not repair and step==22000 and setting=='clean2random':selected=tasks[2:]
                    results=[]
                    for task in selected:
                        source=folder/setting/task;directory=source/'eval_results'/task
                        rows=[];total=0
                        for i in range(50):
                            ok=i%5!=0 if step==22000 else i%4!=0
                            if not repair and step==25000 and setting=='clean2random' and task in tasks[:2]:ok=False
                            total+=ok
                            rows.append(dict(task=task,task_config=scene,instruction_type=lang,instruction=f'{task}-{i}',
                                episode=i,seed=100000+i,inference_seed=inference_seed(scene,task,100000+i),success=ok,
                                cumulative_successes=total,video_recording=False,scene_repair_version=1,initial_state_verified=True))
                        directory.mkdir(parents=True)
                        (directory/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
                        (directory/'_result.txt').write_text(f'Instruction Type: {lang}\n{total/50}\n')
                        results.append(dict(task=task,source=str(source),validated=True,episodes=50,successes=total))
                    status['settings'][setting]=dict(results=results)
                write(folder/'status.json',status)
                if repair:
                    for task in tasks[:2]:
                        write(folder/'paired_episodes/demo_randomized'/f'{task}.json',
                            [dict(episode=i,replay_state=dict(expert_initial_state=True))for i in range(50)])
        out=aggregate(self.root);report=json.loads((out/'comparison.json').read_text())
        self.assertAlmostEqual(report['comparison']['clean2random']['step25000_rate'],.74)
        self.assertAlmostEqual(report['comparison']['equal_weight']['step22000_rate'],.8)
        self.assertTrue((out/'对比汇总.md').exists());self.assertEqual(len(report['provenance']),200)
        # Original buggy scores remain intact, and the final report is immutable.
        self.assertEqual(json.loads((self.root/'base25/status.json').read_text())['settings']['clean2random']['results'][0]['successes'],0)
        with self.assertRaisesRegex(AssertionError,'Never overwrite'):aggregate(self.root)


if __name__=='__main__':unittest.main(verbosity=2)
