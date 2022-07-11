import os
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import OmegaConf


os.environ["WANDB_DISABLED"] = "true"

log = logging.getLogger()


def main_decorator(func):
    def run_script(config):
        # Set the working directory
        if HydraConfig.initialized():
            auto_generated_dir = os.getcwd()
            os.chdir(hydra.utils.get_original_cwd())
        else:
            auto_generated_dir = config.output_dir
        log.info(f"Work dir: {auto_generated_dir}")
        # Save config into yaml format
        with open(Path(auto_generated_dir) / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(config))
        # Enable offline mode for HuggingFace libraries if necessary
        if config.offline_mode:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        os.environ["PYTHONHASHSEED"] = str(config.seed)
        from transformers import set_seed

        set_seed(config.seed)

        func(config, work_dir=Path(auto_generated_dir))

    return run_script
