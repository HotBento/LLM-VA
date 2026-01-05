import os
from argparse import ArgumentParser
import yaml
import json

from dataclasses import dataclass, field
from typing import Optional, Tuple

def add_arg(parser:ArgumentParser, *args, **kwargs):
    if args[0] not in parser._option_string_actions:
        parser.add_argument(*args, **kwargs)
    return parser

@dataclass
class BaseConfig():
    log_path:str            = "log"
    debug:bool              = False
    hf_token:str            = ""
    ppid:int                = -1 # Used to identify the process. It is useful when multiple process are running.
    def update(self, args:dict):
        for key, value in args.items():
            if hasattr(self, key) and value != None and not callable(getattr(self, key)):
                setattr(self, key, value)
    
    def save_yaml(self, path:str):
        with open(path, "w") as f:
            yaml.dump(vars(self), f)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({json.dumps(vars(self), indent=4)})"

    @classmethod
    def get_config(cls, args:Optional[dict]=None):
        config = cls()
        if args == None:
            args = config.parse_args()
        config.update(config._from_yaml(args.get("config_path", None)))
        config.update(args)
        return config
    
    def parse_args(self):
        parser = ArgumentParser()
        
        parser = self.add_args(parser)
        
        args = parser.parse_args()
        return vars(args)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser)->ArgumentParser:
        parser = add_arg(parser, "--config_path", type=str, default=None)
        parser = add_arg(parser, "--log_path", type=str, default=None)
        parser = add_arg(parser, "--debug", action="store_true")
        parser = add_arg(parser, "--hf_token", type=str, default=None)
        parser = add_arg(parser, "--ppid", type=int, default=None)
        
        return parser
    
    def _from_yaml(self, path:str)->dict:
        if path == None:
            return dict()
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def from_yaml(cls, path:str):
        config = cls()
        config.update(config._from_yaml(path))
        return config
    
    @classmethod
    def from_json(cls, path:str):
        config = cls()
        with open(path, "r") as f:
            config.update(json.load(f))
        return config

@dataclass
class ServerConfig(BaseConfig):
    host:str            = "127.0.0.1"
    port:int            = 4397
    model_path:str      = "model_finetuning/lora/qwen"
    model_name:str      = "qwen"
    
    @classmethod
    def add_args(cls, parser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--host", type=str, default=None)
        parser = add_arg(parser, "--port", type=int, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--model_name", type=str, default=None)
        
        return parser

@dataclass
class LVSConfig(BaseConfig):
    model_type:str          = "llama3"
    model_path:str          = "meta-llama/Llama-3.1-8B-Instruct"
    result_path:str         = "result/lvs"
    figure_path:str         = "figure/lvs"
    batch_size:int          = 16
    server_url:str          = "127.0.0.1"
    server_port:int         = 4397
    max_iter:int            = 10
    n_samples:int           = 128
    start_layer:int         = 0
    end_layer:int           = 10000 # End layer is exclusive
    select_token_num:int    = 1 # Number of tokens to select from the end of the sequence
    select_layer_num:int    = 24
    save_hf:bool            = False
    kill_server:bool        = False # Whether to kill the server after running
    add_generation_prompt:bool = True # Whether to add generation prompt during evaluation
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--server_url", type=str, default=None)
        parser = add_arg(parser, "--server_port", type=int, default=None)
        parser = add_arg(parser, "--max_iter", type=int, default=None)
        parser = add_arg(parser, "--n_samples", type=int, default=None)
        parser = add_arg(parser, "--start_layer", type=int, default=None)
        parser = add_arg(parser, "--end_layer", type=int, default=None)
        parser = add_arg(parser, "--select_token_num", type=int, default=None)
        parser = add_arg(parser, "--select_layer_num", type=int, default=None)
        parser = add_arg(parser, "--figure_path", type=str, default=None)
        parser = add_arg(parser, "--save_hf", action="store_true")
        parser = add_arg(parser, "--kill_server", action="store_true")
        parser = add_arg(parser, "--add_generation_prompt", type=bool, default=None)

        return parser
    
class GeneralAbilityConfig(BaseConfig):
    dataset_name:str    = "aime"
    model_type:str      = "llama3"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"
    result_path:str     = "result/general_ability"
    modified:bool       = False
    modify_path:str     = ""
    batch_size:int      = 16
    do_sample:bool      = False
    suffix:str          = "modified"
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_name", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--modified", action="store_true")
        parser = add_arg(parser, "--modify_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--do_sample", action="store_true")
        parser = add_arg(parser, "--suffix", type=str, default=None)

        return parser
