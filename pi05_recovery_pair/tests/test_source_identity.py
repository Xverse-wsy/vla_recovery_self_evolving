import json
from pathlib import Path
import shutil
import subprocess
import pytest
from recovery_pair.source_identity import verify_source_tree
from recovery_pair.data import manifest_autonomous


def test_installed_source_copy_verified_against_git_and_modified_copy_rejected(tmp_path):
    repo=tmp_path/'repo'
    source=repo/'rlinf'
    (source/'models/embodiment/openpi').mkdir(parents=True)
    (source/'models/embodiment/openpi/pi0.py').write_text('PINNED = True\n')
    (source/'config.py').write_text('CONFIG = True\n')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@invalid',
                    '-c','commit.gpgsign=false','commit','-qm','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    installed=tmp_path/'site-packages/rlinf'
    shutil.copytree(source,installed)
    report=verify_source_tree(installed,repo,commit)
    assert report['files_checked']==2
    (installed/'models/embodiment/openpi/pi0.py').write_text('PINNED = False\n')
    with pytest.raises(RuntimeError,match='differs'):
        verify_source_tree(installed,repo,commit)


def test_autonomous_manifest_uses_canonical_task_and_keeps_local_quality_unknown(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=tmp_path/'data'
    (root/'meta').mkdir(parents=True)
    (root/'episode_000000.parquet').parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table({'state':[[0.]*8]*2,'actions':[[0.]*7]*2}),root/'episode_000000.parquet')
    (root/'meta/info.json').write_text(json.dumps({'data_path':'episode_{episode_index:06d}.parquet','chunks_size':1000}))
    provenance=dict(exported_episode_index=0,canonical_task_index=7,suite='libero_10',instruction='put block',
        source_record={'original_episode':'original/parent','used_capx':False,'success':True,
                       'action_dim':7,'trial_id':0,'task_id':1})
    (root/'meta/provenance.jsonl').write_text(json.dumps(provenance)+'\n')
    manifest=manifest_autonomous(root)
    assert manifest[0]['task_id']=='libero_10/task07'
    assert manifest[0]['episode_id']=='original/parent'
    assert manifest[0]['annotations']==[]
