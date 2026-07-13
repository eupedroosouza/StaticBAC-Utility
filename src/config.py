import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

import enums
import sources
import utils


@dataclass
class ModelConfig:
    name: str
    source: sources.Source
    quantized: bool
    inference: enums.InferenceType
    weights: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.source, str):
            self.source = sources.Source[self.source]
        if isinstance(self.inference, str):
            self.inference = enums.InferenceType[self.inference]


@dataclass
class UtilityConfig:
    output_dir: str

    def output_dir(self):
        return Path(self.output_dir)


@dataclass
class StaticBacConfig:
    folder: str
    executable: str


@dataclass
class DatasetConfig:
    image_net: str


class Config:
    def __init__(self, utility: UtilityConfig, staticbac: StaticBacConfig, models: Dict[str, ModelConfig],
                 dataset: DatasetConfig):
        self.utility = utility
        self.staticbac = staticbac
        self.models = models
        self.dataset = dataset


_instance: Optional[Config] = None


def get_config():
    if _instance is None:
        utils.report_error("Configuration requested before initialization.", "Internal Logic Error")
        sys.exit(1)
    return _instance


def load_configuration():
    config_path = Path(utils.root()) / "config.yml"
    try:
        with open(config_path, 'r', encoding="utf-8") as file:
            configuration = yaml.safe_load(file) or {}

            utility = UtilityConfig(**configuration["utility"])
            staticbac = StaticBacConfig(**configuration["static_bac"])

            models_configuration: Dict[str, ModelConfig] = {}
            if "models" in configuration:
                for model_id, model_configuration in configuration["models"].items():
                    models_configuration[model_id] = ModelConfig(**model_configuration)

            dataset = DatasetConfig(**configuration["dataset"])

            global _instance
            _instance = Config(utility=utility, staticbac=staticbac, models=models_configuration, dataset=dataset)

            utils.console.print("[bold white on green] CONFIGURATION [/bold white on green] Loaded configuration.",
                                style="bold green")
            return 1
    except FileNotFoundError:
        utils.report_error(f"Missing configuration file at: {config_path}", "File Not Found",
                           "Check if your project root contains the config.yml file. If not, try pull or recreate that repository.")
    except (KeyError, TypeError) as exc:
        utils.report_error(f"Invalid Schema: Missing required field {exc}", "Configuration Schema Error",
                           "Check your config.yml for missing keys or syntax errors. ")
    except yaml.YAMLError as exc:
        utils.report_error(f"Failed to parse YAML syntax:\n{exc}", "Syntax Error",
                           "Check your config.yml for missing keys or syntax errors. ")
    except Exception as exc:
        utils.report_error(str(exc), "Unexpected Runtime Error")

    return 0
