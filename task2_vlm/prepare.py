import argparse
from common.experiment import read_json
from task2_vlm.data import cache_features, prepare_manifests

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/task2/vlm_196.json'); parser.add_argument('--cache-features', action='store_true'); parser.add_argument('--features-only', action='store_true'); parser.add_argument('--feature-batch-size', type=int, default=32); parser.add_argument('--feature-scope', choices=('all','dev','final'), default='all')
    args = parser.parse_args(); cfg = read_json(args.config)
    if not args.features_only: prepare_manifests(cfg)
    if args.cache_features: cache_features(cfg, args.feature_batch_size, args.feature_scope)
if __name__ == '__main__': main()
