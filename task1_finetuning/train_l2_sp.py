"""Task 1.2: Full fine-tuning with the starting-point penalty lambda=1e-3."""

from task1_finetuning.train_full_ft import main


if __name__ == "__main__":
    main(default_config="configs/task1/full_ft_l2_sp.json", expected_method="full_ft_l2_sp")
