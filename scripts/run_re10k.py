#!/usr/bin/env python3
"""Train or evaluate the released DDN–DepthSplat model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'eval'], default='eval')
    p.add_argument('--variant', choices=['A1', 'A2', 'A3'], default='A3')
    p.add_argument('--arch', help='Override the selected genotype, e.g. for a flat-search candidate')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='Complete released NVS checkpoint; required for evaluation')
    p.add_argument('--student-checkpoint', type=Path, help='Monocular student initialization; required for training')
    p.add_argument('--gmflow-checkpoint', type=Path, help='GMFlow initialization; required for training')
    p.add_argument('--evaluation-index', type=Path, default=ROOT/'third_party/depthsplat/assets/evaluation_index_re10k.json')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/re10k')
    p.add_argument('--batch-size', type=int, default=2, help='Per-GPU training batch size')
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--steps', type=int, default=300001)
    p.add_argument('--test-chunk-interval', type=int, default=10, help='Test-set evaluation chunk interval; set to 1 to evaluate every chunk')
    a = p.parse_args()
    required = [a.data_root, a.evaluation_index]
    if a.mode == 'eval':
        if a.checkpoint is None:
            p.error('--checkpoint is required for evaluation')
        required.append(a.checkpoint)
    else:
        if a.student_checkpoint is None or a.gmflow_checkpoint is None:
            p.error('training requires --student-checkpoint and --gmflow-checkpoint')
        required += [a.student_checkpoint, a.gmflow_checkpoint]
    for path in required:
        if not path.exists():
            p.error(f'File or directory not found: {path}')
    arch = a.arch or json.loads((ROOT/'configs/architectures.json').read_text())[a.variant]
    options = {
        'mode': 'test' if a.mode == 'eval' else 'train',
        'dataset.roots': [str(a.data_root.resolve())],
        'dataset.test_chunk_interval': a.test_chunk_interval,
        'dataset.skip_bad_shape': False,
        'model.encoder.num_scales': 2,
        'model.encoder.upsample_factor': 2,
        'model.encoder.lowest_feature_resolution': 4,
        'model.encoder.monodepth_vit_type': 'vits',
        'model.encoder.student_mono': True,
        'model.encoder.student_mono_arch': arch,
        'model.encoder.student_mono_freeze': a.mode == 'eval',
        'model.encoder.student_mono_pretrained_encoder': False,
        'checkpointing.no_strict_load': False,
        'wandb.mode': 'disabled',
        'trainer.precision': '32-true',
        'output_dir': str(a.output_dir.resolve()),
    }
    command = [sys.executable, '-m', 'src.main', '+experiment=re10k']
    if a.mode == 'eval':
        command += ['dataset/view_sampler=evaluation']
        options.update({
            'dataset.view_sampler.index_path': str(a.evaluation_index.resolve()),
            'checkpointing.pretrained_model': str(a.checkpoint.resolve()),
            'test.compute_scores': True,
        })
    else:
        options.update({
            'model.encoder.student_mono_ckpt': str(a.student_checkpoint.resolve()),
            'model.encoder.student_mono_strict_checkpoint': True,
            'checkpointing.pretrained_mvdepth': str(a.gmflow_checkpoint.resolve()),
            'train.eval_index_path': str(a.evaluation_index.resolve()),
            'data_loader.train.batch_size': a.batch_size,
            'trainer.accumulate_grad_batches': a.accumulate,
            'trainer.max_steps': a.steps,
        })
    command += [f'{key}={json.dumps(value)}' for key, value in options.items()]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, cwd=ROOT/'third_party/depthsplat', env=env, check=True)


if __name__ == '__main__':
    main()
