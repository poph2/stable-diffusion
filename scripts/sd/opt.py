from dataclasses import dataclass


@dataclass
class Opt:
    prompt: str = "a painting of a virus monster playing guitar"
    outdir: str = "outputs/txt2img-samples"
    skip_grid: bool = True
    skip_save: bool = True
    ddim_steps: int = 50
    plms: bool = True
    dpm_solver: bool = True
    laion400m: bool = True
    fixed_code: bool = True
    ddim_eta: float = 0.0
    n_iter: int = 2
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downsampling_factor: int = 8
    n_samples: int = 3
    n_rows: int = 0
    scale: float = 7.5
    from_file: str = ""
    config: str = "configs/stable-diffusion/v1-inference.yaml"
    ckpt: str = "models/ldm/stable-diffusion-v1/model.ckpt"
    seed: int = 42
    precision: str = "autocast"  # ["full", "autocast"]


@dataclass
class ForwardOpt:
    prompt: str = "a painting of a virus monster playing guitar"
    outdir: str = "outputs/txt2img-samples"
    skip_grid: bool = True
    skip_save: bool = True
    ddim_steps: int = 50
    plms: bool = True
    dpm_solver: bool = True
    laion400m: bool = True
    fixed_code: bool = True
    ddim_eta: float = 0.0
    n_iter: int = 2
    height: int = 512
    width: int = 512
    latent_channels: int = 4
    downsampling_factor: int = 8
    n_samples: int = 3
    n_rows: int = 0
    scale: float = 7.5
    from_file: str = ""
    config: str = "configs/stable-diffusion/v1-inference.yaml"
    ckpt: str = "models/ldm/stable-diffusion-v1/model.ckpt"
    seed: int = 42
    precision: str = "autocast"  # ["full", "autocast"]