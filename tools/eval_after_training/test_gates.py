import json
from pathlib import Path
import queue
import tempfile
import unittest

from gates import training_ready,freeze_manifest,take_ready_task


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.plan=dict(source_training_status=str(self.root/'training_status.json'),
            source_training_exit_code=str(self.root/'exit.txt'),source_training_validation=str(self.root/'validation.json'),
            checkpoint=str(self.root/'checkpoint/hf_ckpt'),checkpoint_step=25000,checkpoint_world_size=2,
            manifest_source_directory=str(self.root/'source'),manifest_directory=str(self.root/'frozen'),
            manifest_source_status=str(self.root/'producer_status.json'),episodes_per_task=2)
        self.setting=dict(task_config='demo_clean',instruction_type='seen')

    def tearDown(self):self.temp.cleanup()

    def write(self,name,data):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path

    def rows(self):return [dict(episode=i,seed=100000+i,expert_info={},instructions=dict(seen=f's{i}',unseen=f'u{i}'),initial_actors={'0:actor':[0,0,0,1,0,0,0]}) for i in range(2)]

    def test_training_gate_requires_completion_exit_and_all_shards(self):
        self.assertFalse(training_ready(self.plan)[0])
        self.write('training_status.json',dict(phase='training_to_25000',last_step=24999));self.assertFalse(training_ready(self.plan)[0])
        self.write('training_status.json',dict(phase='complete',final_step=25000));self.assertFalse(training_ready(self.plan)[0])
        (self.root/'exit.txt').write_text('0\n');self.write('validation.json',dict(passed=True,final_step=25000))
        self.assertFalse(training_ready(self.plan)[0])
        for name in ['model/.metadata','optimizer/.metadata','extra_state/extra_state_rank_0.pt','extra_state/extra_state_rank_1.pt']:
            path=self.root/'checkpoint'/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
        self.write('checkpoint/hf_ckpt/model.safetensors.index.json',{'weight_map':{str(i):f'shard{i}' for i in range(6)}})
        for i in range(5):(self.root/'checkpoint/hf_ckpt'/f'shard{i}').touch()
        self.assertFalse(training_ready(self.plan)[0])
        (self.root/'checkpoint/hf_ckpt/shard5').touch();self.assertTrue(training_ready(self.plan)[0])
        # An older checkpoint may be evaluated after the final training checkpoint.
        self.plan.update(checkpoint_step=22000,training_final_step=25000,
            preceding_evaluation_status=str(self.root/'previous.json'),
            preceding_evaluation_exit_code=str(self.root/'previous_exit.txt'))
        self.assertFalse(training_ready(self.plan)[0])
        self.write('previous.json',dict(phase='complete'))
        self.assertFalse(training_ready(self.plan)[0])
        (self.root/'previous_exit.txt').write_text('0\n')
        self.assertTrue(training_ready(self.plan)[0])
        self.write('previous.json',dict(phase='failed'))
        with self.assertRaisesRegex(RuntimeError,'Preceding evaluation failed'):training_ready(self.plan)

    def test_failed_training_never_launches_eval(self):
        self.write('training_status.json',dict(phase='failed',failed=True,error='OOM'))
        with self.assertRaisesRegex(RuntimeError,'Training failed'):training_ready(self.plan)

    def test_partial_manifest_waits_and_freeze_cannot_be_replaced(self):
        self.write('source/demo_clean/task.json',self.rows()[:1]);self.assertFalse(freeze_manifest(self.plan,self.setting,'task',self.root))
        rows=self.rows();rows[-1].pop('initial_actors');self.write('source/demo_clean/task.json',rows)
        self.assertFalse(freeze_manifest(self.plan,self.setting,'task',self.root))
        self.write('source/demo_clean/task.json',self.rows());self.assertTrue(freeze_manifest(self.plan,self.setting,'task',self.root))
        self.assertEqual(json.loads((self.root/'frozen/demo_clean/task.json').read_text()),self.rows())
        rows=self.rows();rows[0]['instructions']['seen']='changed';self.write('source/demo_clean/task.json',rows)
        with self.assertRaisesRegex(RuntimeError,'will not overwrite'):freeze_manifest(self.plan,self.setting,'task',self.root)

    def test_ready_task_skips_waiting_tasks_without_losing_them(self):
        tasks=queue.Queue();tasks.put('waiting');tasks.put('ready')
        self.write('source/demo_clean/ready.json',self.rows())
        task,done=take_ready_task(tasks,self.plan,self.setting,self.root)
        self.assertEqual(task,'ready');self.assertFalse(done);self.assertEqual(list(tasks.queue),['waiting'])
        self.assertEqual(take_ready_task(tasks,self.plan,self.setting,self.root),(None,False))
        self.write('producer_status.json',dict(phase='failed'))
        with self.assertRaisesRegex(RuntimeError,'Seed producer stopped'):take_ready_task(tasks,self.plan,self.setting,self.root)

    def test_record_mode_has_no_external_seed_producer_dependency(self):
        self.plan['manifest_mode']='record'
        self.plan.pop('manifest_source_directory');self.plan.pop('manifest_source_status')
        tasks=queue.Queue();tasks.put('new_task')
        self.assertEqual(take_ready_task(tasks,self.plan,self.setting,self.root),('new_task',False))


if __name__=='__main__':unittest.main(verbosity=2)
