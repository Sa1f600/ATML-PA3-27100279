"""Task 1.4: OPLoRA r=16, alpha=32, protection rank k=128."""

from task1_finetuning.train_full_ft import main


if __name__ == "__main__":
    main(default_config="configs/task1/oplora_16.json", expected_method="oplora")
