"""Non-training checks for Task 2 before an expensive Colab run."""
import argparse
from pathlib import Path
import torch
from common.experiment import read_json
from task2_vlm.data import read_json as read_manifest

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/task2/vlm_196.json'); parser.add_argument('--require-features',action='store_true'); args=parser.parse_args(); cfg=read_json(args.config)
    if not torch.cuda.is_available(): raise RuntimeError('Enable a CUDA GPU runtime in Colab.')
    manifest=read_manifest(Path(cfg['prepared_dir'])/'manifest.json')
    if len(manifest['caption_train']) != 18000 or len(manifest['caption_dev']) != 96: raise RuntimeError('Caption split sizes are incorrect.')
    if len(manifest['natural_train']) != 3072 or len(manifest['natural_dev']) != 64 or len(manifest['natural_final']) != 192: raise RuntimeError('Natural VQA split sizes are incorrect.')
    if len(manifest['annotation_dev']) != 48 or len(manifest['annotation_final']) != 128: raise RuntimeError('Annotation evaluation split sizes are incorrect.')
    all_rows=[row for value in manifest.values() if isinstance(value,list) for row in value]
    if args.require_features:
        missing=[row['image_id'] for row in all_rows if not (Path(cfg['feature_dir'])/f"{row['image_id']}.pt").is_file()]
        if missing: raise RuntimeError(f'Missing {len(set(missing))} cached visual features.')
    print(f"PASS Task 2 preflight: CUDA={torch.cuda.get_device_name(0)}, rows={len(all_rows):,}, features={'checked' if args.require_features else 'not checked'}")
if __name__=='__main__': main()
