from dataclasses import dataclass

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from notebook_helpers import load_model_from_config


def main():
    opt = Opt()

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")


if __name__ == '__main__':
    main()
