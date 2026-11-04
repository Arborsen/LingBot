"""Audit the supplied 2500-episode export before assigning robot task identities."""
import collections
import hashlib
import json
from pathlib import Path
import re


# Exported instructions with known generic/omitted phrases, reviewed against the
# surrounding 50-demo block and the named task's setup/play_once implementation.
# Exact strings prevent applying these exceptions to a different dataset export.
REVIEWED = {
    9: ('adjust_bottle', 'Lift the smooth bottle with red lid from the table.'),
    39: ('adjust_bottle', 'Lift the dark drink bottle with logo up from the table'),
    1528: ('blocks_ranking_size', 'Move small block, then medium block, then large block to their center positions.'),
    1753: ('place_dual_shoes', 'Place both items matching the white and black combination shoe in the orange Nike shoe-box with logo, tip left when set.'),
    1761: ('place_dual_shoes', 'Place the two black and white shoe inside the Nike-branded orange shoe-box, make tips face left.'),
    1774: ('place_dual_shoes', 'Place both items matching the red shoe rounded at front into the orange cardboard shoe-box, ensure shoe tips face left.'),
    1920: ('shake_bottle_horizontally', 'Shake the bottle with white neck and red top horizontally.'),
    1935: ('shake_bottle_horizontally', 'Grab and shake the dark drink bottle with logo horizontally. the right arm.'),
    2151: ('pick_dual_bottles', 'Catch the red-capped bottle, then catch the plastic bottle.'),
    2164: ('pick_dual_bottles', 'Catch the narrow-necked bottle, then catch the bottle with ridges near base.'),
    2171: ('pick_dual_bottles', 'Catch the red-capped bottle, then catch the medium green bottle with capped neck.'),
    2174: ('pick_dual_bottles', 'Catch the red-capped bottle, then catch the bottle with white printed label.'),
}


def build(root, output):
    import pyarrow.parquet as pq
    root, output = Path(root), Path(output)
    metadata = root/'datasets/RoboTwin_lerobot_v30/meta'
    info = json.loads((metadata/'info.json').read_text())
    assert (info['total_episodes'], info['total_frames']) == (2500, 548893)
    expected = set(json.loads((root/'reproduce_artifacts/robotwin_eval_15000_vs10000_50tasks_50episodes_novideo_gpu56_20261017_105400/step15000/task_list.json').read_text()))
    regexes = []
    hashes = {}
    for task in sorted(expected):
        p = root/'RoboTwin/description/task_instruction'/f'{task}.json'
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for template in json.loads(p.read_text()).get('seen', []):
            pattern = ''.join('.+?' if s.startswith('{') and s.endswith('}') else re.escape(s)
                              for s in re.split(r'(\{[^{}]+\})', template))
            regexes.append((task, re.compile('^'+pattern+'$', re.I)))
    episodes = []
    for p in sorted((metadata/'episodes').rglob('*.parquet')):
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        episodes.extend(pq.read_table(p, columns=['episode_index', 'tasks', 'length',
                        'dataset_from_index', 'dataset_to_index']).to_pylist())
    episodes.sort(key=lambda x:x['episode_index'])
    assert [e['episode_index'] for e in episodes] == list(range(2500))
    tasks, rows, exceptions = {}, [], []
    end = 0
    for offset in range(0, 2500, 50):
        block = episodes[offset:offset+50]
        matches = [{t for t,r in regexes if any(r.fullmatch(s) for s in e['tasks'])} for e in block]
        anchors = collections.Counter(t for m in matches if len(m)==1 for t in m)
        task, votes = anchors.most_common(1)[0]
        other = anchors.most_common(2)[1][1] if len(anchors)>1 else 0
        assert votes >= 20 and votes >= other*4 and task not in tasks, (offset,anchors)
        intervals = []
        for e,m in zip(block,matches):
            if task not in m:
                assert REVIEWED.get(e['episode_index']) == (task,e['tasks'][0]), (e,m,task)
                exceptions.append(dict(episode=e['episode_index'], task=task, instruction=e['tasks'][0],
                                       reason='Reviewed literal instruction, task code and contiguous demo block'))
            start, stop = e['dataset_from_index'], e['dataset_to_index']
            assert start == end and stop-start == e['length']
            end = stop
            intervals.append([start,stop])
            rows.append(dict(episode=e['episode_index'],task=task,start=start,end=stop,
                             candidate_tasks=sorted(m),block_unique_anchors=votes))
        tasks[task] = dict(episodes=50,frames=sum(b-a for a,b in intervals), intervals=intervals,
                           unique_instruction_anchors=votes,other_unique_anchors=other)
    assert set(tasks) == expected and end == 548893 and len(exceptions) == len(REVIEWED)
    result = dict(schema_version=1, dataset=str(metadata.parent), episodes=2500, frames=end,
                  policy='50-demo contiguous blocks cross-checked with seen templates and reviewed exceptions',
                  task_index_is_language_id=True, tasks=tasks, episodes_to_tasks=rows,
                  reviewed_exceptions=exceptions,source_sha256=hashes)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(passed=True, episodes=2500,tasks=50,frames=end,reviewed_exceptions=len(exceptions),
                mapping_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
