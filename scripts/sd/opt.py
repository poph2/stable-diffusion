from dataclasses import dataclass


@dataclass
class ModelOpt:
    outdir: str = "outputs/txt2img-samples"
    skip_grid: bool = False
    skip_save: bool = False
    ddim_steps: int = 50
    plms: bool = False
    dpm_solver: bool = False
    laion400m: bool = False
    fixed_code: bool = False
    ddim_eta: float = 0.0
    n_iter: int = 2
    H: int = 512
    W: int = 512
    C: int = 4
    f: int = 8
    n_samples: int = 3
    n_rows: int = 0
    scale: float = 7.5
    from_file: str = ""
    config: str = "../../configs/stable-diffusion/v1-inference.yaml"
    ckpt: str = "models/ldm/stable-diffusion-v1/model.ckpt"
    seed: int = 42
    precision: str = "autocast"  # ["full", "autocast"]

@dataclass
class Text2ImageRequest:
    requestId: str
    requestedAt: str
    prompt: str
