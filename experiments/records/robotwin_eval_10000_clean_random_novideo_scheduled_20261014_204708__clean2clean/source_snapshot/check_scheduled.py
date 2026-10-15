"""CPU checks for scheduling gates, explicit instructions and the no-video episode ledger."""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
import run_scheduled as runner

HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_assets_and_training_gate(self):
        runner.check_assets()
        with mock.patch.object(runner, 'alive', return_value=True):
            ready, detail = runner.readiness()
        self.assertFalse(ready)
        self.assertEqual(detail['reason'], 'training_not_complete')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'status.json').write_text(json.dumps({'phase': 'failed', 'error': 'test failure'}))
            plan = {**runner.PLAN, 'train_root': str(root)}
            with mock.patch.object(runner, 'PLAN', plan), self.assertRaisesRegex(RuntimeError, 'Training failed'):
                runner.readiness()
            (root / 'status.json').write_text(json.dumps({'phase': 'complete', 'final_step': 10000}))
            (root / 'exit_code.txt').write_text('0')
            (root / 'training').mkdir()
            (root / 'training/exit_code.txt').write_text('0')
            plan['checkpoint'] = str(root / 'missing/hf_ckpt')
            with mock.patch.object(runner, 'PLAN', plan), self.assertRaisesRegex(RuntimeError, 'checkpoint incomplete'):
                runner.readiness()

    def test_no_video_client_and_ledgers(self):
        tree = ast.parse((HERE / 'client_instrumented.py').read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name in ('parse_args_and_config', 'eval_policy')]
        namespace = dict(argparse=__import__('argparse'), yaml=yaml, json=json, time=__import__('time'),
            os=os, Path=Path, np=np, UnStableError=type('UnStableError', (Exception,), {}),
            generate_episode_descriptions=lambda *a: [{'seen':['seen test'], 'unseen':['unseen test']}],
            subprocess=mock.Mock())
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<isolated functions>', 'exec'), namespace)
        class Task:
            plan_success = True
            eval_video_path = None
            render_freq = 0
            step_lim = 10
            observations = 0
            actions = 0
            def setup_demo(self, **kwargs):
                self.take_action_cnt = 0
                self.eval_success = False
            def play_once(self):
                return {'info': {}}
            def check_success(self):
                return True
            def close_env(self, **kwargs):
                pass
            def set_instruction(self, instruction):
                self.instruction = instruction
            def get_instruction(self):
                return self.instruction
            def get_obs(self):
                self.observations += 1
                return {'observation': {name: {'rgb':np.zeros((1,1,3),dtype=np.uint8)} for name in
                    ('head_camera','left_camera','right_camera')}, 'joint_action': {'vector':np.zeros(14)}}
            def take_action(self, action):
                self.take_action_cnt += 1
                self.actions += 1
                self.eval_success = self.take_action_cnt == 2
        model = mock.Mock()
        model.infer.return_value = {'action':np.zeros((2,14)), 'server_timing':0.01}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            task = 'lift_pot'
            (source / 'eval_results' / task).mkdir(parents=True)
            for setting in runner.PLAN['settings']:
                command = runner.client_command(task, setting, runner.PLAN['slots'][0], source)
                with mock.patch.object(sys, 'argv', command[2:]):
                    parsed = namespace['parse_args_and_config']()
                self.assertIs(parsed['eval_video_log'], False)
                self.assertEqual(parsed['instruction_type'], setting['instruction_type'])
            setting = runner.PLAN['settings'][0]
            args = dict(task_name=task,policy_name='ACT',task_config='demo_clean',ckpt_setting=None,
                        clear_cache_freq=5,render_freq=0,eval_video_log=False)
            environment = Task()
            with contextlib.redirect_stdout(io.StringIO()):
                namespace['eval_policy'](task,environment,args,model,100000,test_num=10,
                    instruction_type='seen',usr_args={'output_dir':str(source/'eval_results'),'robo_name':'robotwin'})
            self.assertEqual(environment.observations,10)
            self.assertEqual(environment.actions,20)
            namespace['subprocess'].Popen.assert_not_called()
            result = source / 'eval_results' / task / '_result.txt'
            result.write_text('Instruction Type: seen\n\n1.0')
            self.assertTrue(runner.validate_task(task,setting,source,0,1)['validated'])
            ledger = result.parent / 'episodes.jsonl'
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[1]['seed'] = rows[0]['seed']
            ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertFalse(runner.validate_task(task,setting,source,0,1)['validated'])
            rows[1]['seed'] = 100001
            rows[0]['instruction_type'] = 'unseen'
            ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertFalse(runner.validate_task(task,setting,source,0,1)['validated'])
        wrapper = (HERE / 'eval_no_video.py').read_text()
        self.assertNotIn('wrap_take_action', wrapper)
        self.assertNotIn('video_recording', wrapper)


if __name__ == '__main__':
    os.chdir(runner.PLAN['sim'])
    unittest.main()
