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
class DatasetConfig(BaseConfig):
    src_path:str                = "dataset/advbench.csv"
    src_type:str                = "AdvBench"
    dst_path:str                = "dataset"
    template_type:list[str]     = field(default_factory=list)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--src_path", type=str, default=None)
        parser = add_arg(parser, "--src_type", type=str, default=None)
        parser = add_arg(parser, "--dst_path", type=str, default=None)
        parser = add_arg(parser, "--template_type", nargs="+", type=str, default=None)
        
        return parser

@dataclass
class EvalConfig(BaseConfig):
    gen_path:str        = "result/result_gen/llama3/advbench-0"
    batch_size:int      = 32
    result_path:str     = "result/result_eval"
    base_model:str      = "llama3"
    eval_model:str      = "meta-llama/Llama-3.3-70B-Instruct"
    eval_key:str        = ""
    base_url:str        = ""
    eval_type:str       = "toxic"
    is_peft:bool        = False
    th:float            = 0.5
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--gen_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--eval_key", type=str, default=None)
        parser = add_arg(parser, "--base_url", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--eval_type", type=str, default=None)
        
        return parser

@dataclass
class GenerateConfig(BaseConfig):
    dataset_path:str    = "dataset/advbench.csv"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"    
    model_type:str      = "llama3"
    dataset_type:str    = "AdvBench"
    batch_size:int      = 256
    max_new_tokens:int  = 128
    result_path:str     = "result/result_gen"
    guard_path:str      = ""
    guard_type:str      = ""
    model_key:str       = ""
    guard_key:str       = ""
    if_modify:bool      = False
    modified_path:str   = "result/result_tuning/llama3-0"
    model_name:str      = "meta-llama/Meta-Llama-3-8B-Instruct"
    finetuning:bool     = False
    

    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--dataset_type", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--max_new_tokens", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--guard_path", type=str, default=None)
        parser = add_arg(parser, "--guard_type", type=str, default=None)
        parser = add_arg(parser, "--model_key", type=str, default=None)
        parser = add_arg(parser, "--guard_key", type=str, default=None)
        parser = add_arg(parser, "--if_modify", action="store_true")
        parser = add_arg(parser, "--modified_path", type=str, default=None)
        parser = add_arg(parser, "--model_name", type=str, default=None)
        parser = add_arg(parser, "--finetuning", action="store_true")
        
        return parser
    
@dataclass
class HeadIdentifyConfig(BaseConfig):
    # dataset_path:str        = "dataset/all-advbench.csv"
    cor_benign_path:str     = "dataset/all-nq.csv"
    cor_toxic_path:str      = "dataset/all-advbench.csv"
    respond_benign_path:str = "dataset/respond-nq.csv"
    respond_toxic_path:str  = "dataset/respond-advbench.csv"
    model_path:str          = "meta-llama/Llama-3.1-8B-Instruct"
    model_name:str          = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_type:str          = "llama3"
    batch_size:int          = 16
    max_new_tokens:int      = 64
    result_path:str         = "result/result_identify"
    k_benign:int            = 5
    k_toxic:int             = 5
    eval_dataset:str        = "dataset/xstest_v2_prompts.csv"
    save:bool               = False
    eval:bool               = False
    
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        # parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--cor_benign_path", type=str, default=None)
        parser = add_arg(parser, "--cor_toxic_path", type=str, default=None)
        parser = add_arg(parser, "--respond_benign_path", type=str, default=None)
        parser = add_arg(parser, "--respond_toxic_path", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--model_name", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--max_new_tokens", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--k_benign", type=int, default=None)
        parser = add_arg(parser, "--k_toxic", type=int, default=None)
        parser = add_arg(parser, "--eval_dataset", type=str, default=None)
        parser = add_arg(parser, "--save", action="store_true")
        parser = add_arg(parser, "--eval", action="store_true")
        
        return parser
    
@dataclass
class ModelModifyConfig(BaseConfig):
    head_path:str       = "result/result_identify/llama3-0/heads.csv"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"
    model_name:str      = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_type:str      = "llama3"
    k_benign:int        = 5
    lambda_benign:float = 1.0
    k_toxic:int         = 5
    lambda_toxic:float  = 1.0
    if_random:bool      = False
    
    # save:bool           = False
    # result_path:str     = ""
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--head_path", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--model_name", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--k_benign", type=int, default=None)
        parser = add_arg(parser, "--lambda_benign", type=float, default=None)
        parser = add_arg(parser, "--k_toxic", type=int, default=None)
        parser = add_arg(parser, "--lambda_toxic", type=float, default=None)
        parser = add_arg(parser, "--if_random", action="store_true")
        # parser = add_arg(parser, "--result_path", type=str, default=None)
        # parser = add_arg(parser, "--save", action="store_true")
        
        return parser
    
@dataclass
class TuningConfig(BaseConfig):
    model_path:str                      = "meta-llama/Llama-3.1-8B-Instruct"
    model_name:str                      = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_type:str                      = "llama3"
    result_path:str                     = "result/result_tuning"
    k_benign_range:Tuple[int]           = (0, 10)
    lambda_benign_range:Tuple[float]    = (1.0, 5.0)
    k_toxic_range:Tuple[int]            = (0, 10)
    lambda_toxic_range:Tuple[float]     = (-1.0, 1.0)
    head_path:str                       = "result/result_identify/llama3-0/heads.csv"
    alpha:float                         = 0.5
    batch_size:int                      = 16
    max_new_tokens:int                  = 128
    n_trials:int                        = 100
    or_dataset:str                      = "result/result_final_100"
    toxic_dataset:str                   = "dataset/sampled_advbench.csv"
    if_random:bool                      = False
    server:str                          = "127.0.0.1"
    port:int                            = 4397
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--model_name", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--k_benign_range", type=int, default=None)
        parser = add_arg(parser, "--lambda_benign_range", type=float, default=None)
        parser = add_arg(parser, "--k_toxic_range", type=int, default=None)
        parser = add_arg(parser, "--lambda_toxic_range", type=float, default=None)
        parser = add_arg(parser, "--head_path", type=str, default=None)
        parser = add_arg(parser, "--alpha", type=float, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--n_trials", type=int, default=None)
        parser = add_arg(parser, "--or_dataset", type=str, default=None)
        parser = add_arg(parser, "--toxic_dataset", type=str, default=None)
        parser = add_arg(parser, "--max_new_tokens", type=int, default=None)
        parser = add_arg(parser, "--if_random", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--port", type=int, default=None)
        
        return parser

# TODO: Close-source LLMs support
@dataclass
class GenConfig(BaseConfig):
    dataset_path:str    = "dataset_training/merged.csv"
    sample_num:int      = 20
    gen_model:str       = "qwen-ds"
    eval_model:str      = "model_finetuning/lora/qwen"
    batch_size:int      = 10
    total_round:int     = 10
    result_path:str     = "result/result_gendata"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    n_memory:int        = 1
    stream:bool         = False
    server:str          = ""
    model_list:list[str] = field(default_factory=list)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--sample_num", type=int, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--total_round", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--model_list", nargs="+", type=str, default=None)
        
        return parser

@dataclass
class EvolutionConfig(BaseConfig):
    dataset_path:str    = "dataset_training/merged.csv"
    sample_num:int      = 20
    gen_model:str       = "qwen-ds"
    eval_model:str      = "model_finetuning/lora/qwen"
    batch_size:int      = 20
    total_round:int     = 5
    result_path:str     = "result/result_evolution"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    n_memory:int        = 1
    stream:bool         = False
    mutate_rate:float   = 0.5
    server:str          = ""
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--sample_num", type=int, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--total_round", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--mutate_rate", type=float, default=None)
        parser = add_arg(parser, "--server", type=str, default=None)
        
        return parser
    
@dataclass
class RLConfig(BaseConfig):
    dataset_path:str    = "dataset_training/merged.csv"
    sample_num:int      = 1
    gen_model:str       = "Qwen/Qwen2.5-14B-Instruct"
    eval_model:str      = "model_finetuning/lora/qwen"
    batch_size:int      = 1000
    total_round:int     = 10
    result_path:str     = "result/result_grpo"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    stream:bool         = False
    provide_example:bool= False
    resume:bool         = False
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--sample_num", type=int, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--total_round", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--provide_example", action="store_true")
        parser = add_arg(parser, "--resume", action="store_true")
        
        return parser

@dataclass
class FullGenConfig(BaseConfig):
    dataset_path:str    = "dataset_training/merged.csv"
    gen_model:str       = "qwen-ds"
    eval_model:str      = "model_finetuning/lora/qwen"
    batch_size:int      = 10
    result_path:str     = "result/result_gendata"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    n_memory:int        = 1
    stream:bool         = False
    server:str          = ""
    model_list:list[str] = field(default_factory=list)
    modified_path:str   = "result/result_tuning/llama3-0"
    finetuning:bool     = False
    if_modify:bool      = False
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--model_list", nargs="+", type=str, default=["llama3", "gemma", "mistral", "phi", "qwen"])
        parser = add_arg(parser, "--modified_path", type=str, default=None)
        parser = add_arg(parser, "--finetuning", action="store_true")
        parser = add_arg(parser, "--if_modify", action="store_true")
        
        return parser

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

class AbliterationConfig(BaseConfig):
    harmful_dataset_path:str    = "dataset/advbench_alter.csv"
    harmless_dataset_path:str   = "result/result_final/llama3_or.csv"
    model_type:str              = "llama3"
    model_path:str              = "meta-llama/Llama-3.1-8B-Instruct"
    result_path:str             = "result/result_ablation"
    
    @classmethod
    def add_args(cls, parser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--harmful_dataset_path", type=str, default=None)
        parser = add_arg(parser, "--harmless_dataset_path", type=str, default=None)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        
        return parser

@dataclass
class ORFuzzConfig(BaseConfig):
    dataset_path:str    = "dataset/full.csv"
    sample_num:int      = 3
    gen_model:str       = "qwen-ds"
    eval_model:str      = "model_finetuning/lora/qwen"
    total_round:int     = 50
    result_path:str     = "result/result_gendata"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    div_rate:float      = 0.5
    n_memory:int        = 5
    mutator_manager:str = "ucb"
    selection_type:str  = "mcts_explore"
    reconstruction:bool = False
    num_seed:int        = 5
    n_mutator:int       = 3
    refiner:str         = "CoT"
    stream:bool         = False
    server:str          = ""
    model_list:list[str] = field(default_factory=list)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--sample_num", type=int, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--total_round", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--div_rate", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--mutator_manager", type=str, default=None)
        parser = add_arg(parser, "--selection_type", type=str, default=None)
        parser = add_arg(parser, "--reconstruction", action="store_true")
        parser = add_arg(parser, "--refiner", type=str, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--model_list", nargs="+", type=str, default=["llama3", "gemma", "mistral", "phi", "qwen"])
        
        return parser

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

class PrepareAlphaConfig(BaseConfig):
    model_type:str      = "llama3"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"
    result_path:str     = "result"
    modified:bool       = False
    modify_path:str     = ""
    batch_size:int      = 16
    do_sample:bool      = False
    server_url:str      = "127.0.0.1"
    server_port:int     = 4397
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--modified", action="store_true")
        parser = add_arg(parser, "--modify_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--do_sample", action="store_true")
        parser = add_arg(parser, "--server_url", type=str, default=None)
        parser = add_arg(parser, "--server_port", type=int, default=None)

        return parser

class ModelEvalConfig(BaseConfig):
    model_type:str      = "llama3"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"
    modify_method:str   = "original"
    modify_path:str     = ""
    result_path:str     = "result"
    batch_size:int      = 16
    server_url:str      = "127.0.0.1"
    server_port:int     = 4397
    do_sample:bool      = False

    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--modify_method", type=str, default=None)
        parser = add_arg(parser, "--modify_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--server_url", type=str, default=None)
        parser = add_arg(parser, "--server_port", type=int, default=None)
        parser = add_arg(parser, "--do_sample", action="store_true")

        return parser

class GenSurveyConfig(BaseConfig):
    model_type:str      = "llama3"
    model_path:str      = "meta-llama/Llama-3.1-8B-Instruct"
    result_path:str     = "result/result_survey"
    batch_size:int      = 16
    max_new_tokens:int  = 128
    server_url:str      = "127.0.0.1"
    server_port:int     = 4397
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--model_type", type=str, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--max_new_tokens", type=int, default=None)
        parser = add_arg(parser, "--server_url", type=str, default=None)
        parser = add_arg(parser, "--server_port", type=int, default=None)

        return parser