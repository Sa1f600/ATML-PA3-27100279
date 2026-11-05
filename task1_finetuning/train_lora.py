"""Task 1.3: independent LoRA rank 4 or rank 16 training runs."""

import argparse

from task1_finetuning.train_full_ft import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, choices=[4, 16], required=True)
    args, remaining = parser.parse_known_args()
    main(default_config=f"configs/task1/lora_{args.rank}.json", expected_method="lora",
         argv=remaining, expected_rank=args.rank)
