import torch
from abc import ABC, abstractmethod
from loguru._logger import Logger
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformer_lens import HookedTransformer
from typing import Optional

OPEN_LLM_TYPE_LIST = ["llama2", "llama3", "gemma", "mistral", "phi", "vicuna", "falcon", "llama-guard", "qwen"]
model_name_dict = {
    "llama3":"meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma":"google/gemma-2-9b-it",
    "mistral":"mistralai/Mistral-7B-Instruct-v0.3",
    "phi":"microsoft/Phi-3.5-mini-instruct",
    "qwen2-3b":"Qwen/Qwen2.5-3B-Instruct",
    "qwen":"Qwen/Qwen2.5-7B-Instruct",
    "qwen2-14b":"Qwen/Qwen2.5-14B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b":"Qwen/Qwen3-14B",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4": "microsoft/phi-4"
}


class BaseModel(ABC):
    def __init__(self, system_prompt:str) -> None:
        self.system_prompt = system_prompt
    
    @abstractmethod
    def generate(self, input:list[str], max_new_tokens:int, logger:Logger|None=None)->list[str]:
        pass
    
class OpenModel(BaseModel):
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, system_prompt:str) -> None:
        super().__init__(system_prompt)
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.history = []
    
    def generate(self, inputs: list[str], max_new_tokens: int, logger: Logger | None = None, history:Optional[list[list]] = None, autocast=False, enable_thinking=False, **kwargs)->list[str]:
        chat = []
        self.history = []
        if history == None or history == []:
            for i in inputs:
                if self.system_prompt != None:
                    chat_list = [
                        {"role" : "system", "content" : self.system_prompt},
                        {"role" : "user", "content" : i},
                    ]
                else:
                    chat_list = [
                        {"role" : "user", "content" : i},
                    ]
                self.history.append(chat_list)
                chat.append(self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking))
        else:
            for i in range(len(inputs)):
                if history[i] != None:
                    chat_list = history[i] + [{"role" : "user", "content" : inputs[i]}]
                else:
                    if self.system_prompt != None:
                        chat_list = [
                            {"role" : "system", "content" : self.system_prompt},
                            {"role" : "user", "content" : inputs[i]},
                        ]
                    else:
                        chat_list = [
                            {"role" : "user", "content" : inputs[i]},
                        ]
                self.history.append(chat_list)
                chat.append(self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True))
        tokenized_inputs = self.tokenizer(chat, return_tensors="pt", padding=True).to(self.device)
        if autocast:
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    gen = self.model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens, **kwargs)
        else:
            gen = self.model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens, **kwargs)
        if logger != None:
            logger.debug(f"gen: {gen}")
        gen = self.tokenizer.batch_decode(gen[:,tokenized_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(len(gen)):
            self.history[i].append({"role" : "assistant", "content" : gen[i]})
        if logger != None:
            logger.debug(f"gen: {gen}\ninputs:{tokenized_inputs}")
        for i in range(len(gen)):
            if logger != None:
                logger.info(f"Input: {chat[i]}")
                logger.info(f"Output: {gen[i]}")
        return gen
    
    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self



def load_model_and_tokenizer(model_name:str, model_path:str, logger:Optional[Logger]=None, device="cuda:0"):
    if "gemma" in model_name.lower() or "gemma" in model_path.lower():
        attn_implementation = "eager"
    elif "phi" in model_name.lower() or "phi" in model_path.lower():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    hf_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation, trust_remote_code=True)
    hf_model.generation_config.disable_compile=True # disable the compile to avoid error in some models
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hf_model.generation_config.pad_token_id == None:
        if isinstance(hf_model.generation_config.eos_token_id, list):
            hf_model.generation_config.pad_token_id = hf_model.generation_config.eos_token_id[0]
        elif isinstance(hf_model.generation_config.eos_token_id, int):
            hf_model.generation_config.pad_token_id = hf_model.generation_config.eos_token_id

    hooked_model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.bfloat16,
        default_padding_side="left",
        default_prepend_bos=False,
        move_to_device=False,
    ).to(device)
    return hf_model, hooked_model, tokenizer
