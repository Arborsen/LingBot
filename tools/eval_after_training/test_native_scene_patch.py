import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


class NativePatchTests(unittest.TestCase):
    def test_patch_is_guarded_repeatable_and_keeps_original_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'sim/envs/_base_task.py';source.parent.mkdir(parents=True)
            original='''class Base:
    table_xy_bias=[0.3,0]
    def get_cluttered_table(self, cluttered_numbers=10, xlim=[-0.59, 0.59], ylim=[-0.34, 0.34], zlim=[0.741]):
        xlim[0]+=self.table_xy_bias[0]
        xlim[1]+=self.table_xy_bias[0]
        return xlim,ylim,zlim
'''
            source.write_text(original);digest=hashlib.sha256(source.read_bytes()).hexdigest()
            cfg={}
            for step in [22000,25000]:
                folder=root/f'step{step}';folder.mkdir();cfg[f'repair{step}']=str(folder)
                (folder/'plan.json').write_text(json.dumps(dict(sim=str(root/'sim'))))
                (folder/'source_hashes.json').write_text(json.dumps({str(source):digest}))
            (root/'campaign.json').write_text(json.dumps(cfg))
            argv=sys.argv;sys.argv=['run_scene_campaign.py',str(root)]
            try:
                spec=importlib.util.spec_from_file_location('patch_test',Path(__file__).with_name('run_scene_campaign.py'))
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            finally:sys.argv=argv
            module.apply_native_fix()
            namespace={};exec(source.read_text(),namespace);env=namespace['Base']()
            calls=[env.get_cluttered_table() for _ in range(120)]
            self.assertTrue(all(row==calls[0] for row in calls))
            self.assertEqual((root/'simulator_source_before.py').read_text(),original)
            for step in [22000,25000]:
                folder=Path(cfg[f'repair{step}'])
                self.assertEqual(json.loads((folder/'source_hashes_before_native_fix.json').read_text())[str(source)],digest)
                self.assertEqual(json.loads((folder/'source_hashes.json').read_text())[str(source)],hashlib.sha256(source.read_bytes()).hexdigest())
            with self.assertRaisesRegex(AssertionError,'reviewed patch'):module.apply_native_fix()


if __name__=='__main__':unittest.main(verbosity=2)
