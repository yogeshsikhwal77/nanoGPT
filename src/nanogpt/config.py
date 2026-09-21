import yaml
from dataclasses import dataclass

#------   Base pre-training configs ----

@dataclass
class ModelConfig:
    dim: int
    layers: int
    heads: int
    context_len: int
    vocab_size: int


@dataclass
class TrainConfig:
    batch_size: int
    grad_accum: int
    lr : float
    epochs: int
    seed : int
    data_dir: str
    save_path: str

@dataclass
class BaseAppConfig:
    model: ModelConfig
    train: TrainConfig


# ---- SFT Config -----

@dataclass
class SFTConfig:
    model_path: str
    data_path: str
    save_path: str
    batch_size: int
    grad_accum: int
    lr: float
    epochs: int


#--- loaders----
def load_base_config(yaml_path: str) -> BaseAppConfig:
    """Loads the nested base.yaml file"""
    with open(yaml_path,'r') as f:
        data = yaml.safe_load(f)

    return BaseAppConfig(
        model = ModelConfig(**data.get('model',{})),
        train = TrainConfig(**data.get('train',{}))
    )

def load_sft_config(yaml_path: str) -> SFTConfig:
    """Loads the nested sft.yaml file"""
    with open(yaml_path,'r') as f:
        data = yaml.safe_load(f)

    return SFTConfig(**data)  # ** is unpacking operator



