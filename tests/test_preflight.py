from common.experiment import read_json
from task1_finetuning import preflight


def test_training_config_uses_shared_harness_pin(monkeypatch):
    """Preflight must accept a training config without evaluation-only keys."""
    versions = {
        "torch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "transformers": "4.53.1",
        "datasets": "3.6.0",
        "lm_eval": "0.4.9.1",
    }
    monkeypatch.setattr(preflight, "package_version", versions.__getitem__)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight.torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(preflight.torch.cuda, "device_count", lambda: 1)

    preflight.check_environment(read_json("configs/task1/full_ft_l2_sp.json"))
