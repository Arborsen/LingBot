"""Check public headline numbers against the exported per-task evidence."""
import csv
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
BENCHMARK=ROOT/'experiments/benchmarks/step22000_step25000'


class PublicResultTests(unittest.TestCase):
    def test_task_rows_reconcile_with_summary_and_completion_scope(self):
        summary=json.loads((BENCHMARK/'summary.json').read_text())
        with (BENCHMARK/'per_task.csv').open(encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),100)
        self.assertEqual(len({(r['task'],r['setting']) for r in rows}),100)
        for step,settings in summary['checkpoints'].items():
            for setting,stats in settings.items():
                selected=[r for r in rows if r['setting']==setting and r[f'step{step}_successes']!='']
                self.assertEqual(sum(int(r[f'step{step}_successes'])for r in selected),stats['successes'])
                self.assertEqual(50*len(selected),stats['validated_episodes'])
                self.assertEqual(len(selected),stats['validated_tasks'])
                self.assertEqual(stats['status']=='complete',len(selected)==50)
                self.assertAlmostEqual(stats['success_rate'],stats['successes']/stats['validated_episodes'])
        self.assertEqual(summary['checkpoints']['22000']['clean2random']['status'],'partial')
        self.assertFalse(summary['protocol']['rtc_enabled'])
        self.assertEqual(summary['checkpoints']['25000']['clean2random']['successes'],1088)

    def test_provenance_has_hashes_without_machine_paths(self):
        evidence=json.loads((BENCHMARK/'provenance.json').read_text())
        for record in evidence['evaluations']:
            self.assertFalse(Path(record['source']).is_absolute())
            for field in ['ledger_sha256','result_sha256']:
                self.assertEqual(len(record[field]),64)
                self.assertTrue(all(c in '0123456789abcdef'for c in record[field]))


if __name__=='__main__':unittest.main(verbosity=2)
