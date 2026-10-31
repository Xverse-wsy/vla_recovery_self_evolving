"""Exercise stage wiring with a command recorder, without a real GPU or updates."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_positive_pipeline_is_staged_and_uses_all_train_with_selected_gpus(tmp_path):
    code = Path(__file__).resolve().parents[1]
    script = code/'scripts/run_positive_fm_pipeline.sh'
    root = tmp_path/'data with spaces'
    checkpoint = root/'base'
    checkpoint.mkdir(parents=True)
    (checkpoint/'model.safetensors').touch()
    norm = root/'norm.json'
    norm.write_text('{}')
    experts=root/'dataset_subsets/rlinf_pi05_libero_fewshot40_20261021'
    experts.mkdir(parents=True)
    (experts/'episodes.json').write_text('[]')
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps({'norm_stats':'norm.json','algorithm_config':'unused'}))
    log = tmp_path/'commands.jsonl'
    wrapper = tmp_path/'record_python'
    wrapper.write_text('#!'+sys.executable+'\n'+r'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args[0] in ('-','-c'):
    os.execv(sys.executable,[sys.executable,*args])
with open(os.environ['RECORD_COMMANDS'],'a') as f:
    f.write(json.dumps(dict(args=args,gpus=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\n')
def option(name):
    return Path(args[args.index(name)+1])
if 'build-libraries' in args:
    p=option('--output')/'windows';p.mkdir(parents=True)
    (p/'windows.jsonl').touch()
elif 'numeric-preview' in args or 'recovery_pair.vision_trial' in args or 'recovery_pair.success_admission' in args:
    option('--output').mkdir(parents=True)
elif 'prepare-positive-fm' in args:
    p=option('--output');p.mkdir(parents=True)
    (p/'report.json').write_text(json.dumps(dict(split_policy='all_train_v1',validation_enabled=False)))
''')
    wrapper.chmod(0o755)
    env = dict(os.environ, PAIR_PYTHON=str(wrapper), PAIR_DATA_ROOT=str(root),
        PAIR_CHECKPOINT=str(checkpoint), PAIR_NORM_STATS=str(norm),
        PAIR_SOURCES=str(sources), PAIR_CONFIG=str(code/'configs/libero_pi05_positive_fm.json'),
        PAIR_RUN=str(tmp_path/'run with spaces'), PAIR_VISION_GPU='2',
        CUDA_VISIBLE_DEVICES='1,3', NPROC_PER_NODE='2',RECORD_COMMANDS=str(log))
    def run(stage):
        subprocess.run(['bash',str(script),stage],env=env,check=True,capture_output=True,text=True)
    def calls():
        return [json.loads(x) for x in log.read_text().splitlines()]
    run('prepare')
    prepared_calls = calls()
    assert len(prepared_calls)==6  # libraries, numeric, vision, admission, prepare, deep audit
    assert not any('train' in c['args'] for c in prepared_calls)
    assert prepared_calls[2]['gpus']=='2'
    assert '--all-train' in prepared_calls[4]['args']
    assert '--expert-dataset' in prepared_calls[4]['args']
    assert prepared_calls[4]['args'][prepared_calls[4]['args'].index('--expert-dataset')+1]==str(experts)
    assert '--expert-dataset' not in prepared_calls[0]['args']  # not in matching banks
    copied=json.loads((Path(env['PAIR_RUN'])/'sources.json').read_text())
    assert copied['algorithm_config']==str(Path(env['PAIR_RUN'])/'algorithm.json')
    assert json.loads(sources.read_text())['algorithm_config']=='unused'
    env['PAIR_MAX_UPDATES']='1'
    run('train')
    multi=calls()[-1]
    assert 'torch.distributed.run' in multi['args'] and '--nproc_per_node=2' in multi['args']
    assert multi['gpus']=='1,3'
    assert multi['args'][multi['args'].index('--mode')+1]=='fm_only'
    assert multi['args'][multi['args'].index('--max-updates')+1]=='1'
    assert '--max-cuda-memory-gb' in multi['args']
    env.update(CUDA_VISIBLE_DEVICES='2',NPROC_PER_NODE='1',PAIR_RESUME='/trusted/update_000001.pt')
    run('train')
    single=calls()[-1]['args']
    assert 'torch.distributed.run' not in single and '--resume' in single
    env['PAIR_EXPORT_CHECKPOINT']='/trusted/update_000100.pt'
    run('export')
    assert calls()[-1]['args']==['-u','-m','recovery_pair','export','--checkpoint',
        env['PAIR_EXPORT_CHECKPOINT'],'--output',str(Path(env['PAIR_RUN'])/'exported')]
    before=len(calls())
    env['NPROC_PER_NODE']='2'
    result=subprocess.run(['bash',str(script),'train'],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and len(calls())==before
