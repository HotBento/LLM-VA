import openai
import anthropic

import httpx
import torch
from abc import ABC, abstractmethod
from loguru._logger import Logger
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM
)
from transformer_lens import HookedTransformer
from typing import Optional

import yaml

from utils.prompt import SYS_PROMPT
import time

# m = LlamaForCausalLM()
# m.generate

OPEN_LLM_TYPE_LIST = ["llama2", "llama3", "gemma", "mistral", "phi", "vicuna", "falcon", "llama-guard", "qwen"]
CLOSE_LLM_TYPE_LIST = ["openai", "claude", "gemini", "chatglm"]
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
        # logger.info(f"tokenized_inputs: {tokenized_inputs}")
        # logger.info(f"self.device: {self.device}")
        # logger.info(f"model.device: {self.model.device}")
        # logger.info(f"tokenizer_input.device: {tokenized_inputs['input_ids'].device}")
        # logger.info(f"chat: {chat}")
        if autocast:
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    gen = self.model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens, after=tokenized_inputs["input_ids"].shape[1]-1, **kwargs)
        else:
            gen = self.model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens, after=tokenized_inputs["input_ids"].shape[1]-1, **kwargs)
        if logger != None:
            logger.debug(f"gen: {gen}")
        gen = self.tokenizer.batch_decode(gen[:,tokenized_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(len(gen)):
            self.history[i].append({"role" : "assistant", "content" : gen[i]})
        if logger != None:
            logger.debug(f"gen: {gen}\ninputs:{tokenized_inputs}")
        for i in range(len(gen)):
            # gen[i] = gen[i]
            if logger != None:
                logger.info(f"Input: {chat[i]}")
                logger.info(f"Output: {gen[i]}")
        return gen
    
    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

# TODO: separate openai api and claude api
class CloseModel(BaseModel):
    def __init__(self, model:str, model_type:str, system_prompt:str, api_key:str, proxies:dict[str,str]|None=None, base_url:Optional[str]=None) -> None:
        super().__init__(system_prompt)
        self.api_key = api_key
        self.model_type = model_type
        httpx_client = httpx.Client(proxy=proxies)
        if base_url == None:
            if model_type.lower() == "openai":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    http_client=httpx_client,
                    base_url=f"https://api.openai.com/v1",
                )
                self.model_name = model
            elif model_type.lower() == "deepseek":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    http_client=httpx_client,
                    base_url=f"https://api.deepseek.com",
                )
                self.model_name = model
            elif model_type.lower() == "baidu-ds":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    http_client=httpx_client,
                    base_url=f"https://qianfan.baidubce.com/v2",
                )
                self.model_name = model
            elif model_type.lower() == "tengxun-ds":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    http_client=httpx_client,
                    base_url=f"https://api.lkeap.cloud.tencent.com/v1",
                )
                self.model_name = model
            elif model_type.lower() == "qwen-ds":
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    http_client=httpx_client,
                    base_url=f"https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                self.model_name = model
            else:
                raise ValueError(f"Unkown model type: {model_type}")
        else:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=httpx_client,
                base_url=base_url,
            )
            self.model_name = model
        
        self.history = []
    
    def generate(self, inputs: list[str], max_new_tokens: int, logger: Logger | None = None, history:Optional[list[list[dict[str, str]]]]=None, retry=200, stream=False, change_history=True) -> list[str]:
        chat = []
        if logger != None:
            logger.info(f"begin")
        if change_history:
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
                if change_history:
                    self.history.append(chat_list)
                chat.append(chat_list)
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
                if change_history:
                    self.history.append(chat_list)
                chat.append(chat_list)
        gen = []
        reason = []
        for i in range(len(inputs)):
            generated_content = ""
            reasoning_content = ""
            for _ in range(retry):
                try:
                    if self.model_type.lower() == "openai" or self.model_type.lower() == "deepseek" or "ds" in self.model_type.lower():
                        if stream:
                            result = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=chat[i],
                                timeout=300,
                                max_tokens=max_new_tokens,
                                stream=True
                            ) 
                            reasoning_content = ""
                            generated_content = ""
                            # first_chunk = True
                            for chunk in result:
                                # if first_chunk:
                                    # logger.info(chunk)
                                    # first_chunk = False
                                if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
                                    reasoning_content += chunk.choices[0].delta.reasoning_content
                                    # print(chunk.choices[0].delta.reasoning_content, end="")
                                elif chunk.choices[0].delta.content:
                                    generated_content += chunk.choices[0].delta.content
                                    # print(chunk.choices[0].delta.content, end="")
                            
                        else:
                            result = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=chat[i],
                                timeout=300,
                                max_tokens=max_new_tokens
                            ) 
                            generated_content = result.choices[0].message.content
                            if hasattr(result.choices[0].message, "reasoning_content") and "</think>" not in generated_content:
                                reasoning_content = result.choices[0].message.reasoning_content
                            else:
                                reasoning_content = ""
                            #     generated_content = "<think>" + result.choices[0].message.reasoning_content + "</think>" + generated_content
                    elif self.model_type.lower() == "claude":
                        # TODO, stream
                        result = self.client.messages.create(
                            model=self.model_name,
                            system=self.system_prompt,
                            messages=chat[i],
                            timeout=1000,
                            max_tokens=max_new_tokens
                        )
                        generated_content = result.content[0]["text"]
                    else:
                        raise ValueError(f"Unkown model type: {self.model_type}")

                    break
                except Exception as e:
                    if logger != None:
                        logger.info(f'chat: {chat[i]}')
                        logger.info('error; waiting 10 seconds and resubmitting')
                        logger.error(e)
                    time.sleep(10)
            gen.append(generated_content)
            reason.append(reasoning_content)
        for i in range(len(gen)):
            if change_history:
                self.history[i].append({"role" : "assistant", "content" : "<think>" + reason[i]+"\n</think>\n"+gen[i]})
            # if logger != None:
            #     # logger.info(f"Input: {inputs[i]}")
            #     logger.info(f"Reasoning: {reason[i]}")
            #     logger.info(f"Output: {gen[i]}")
        return gen

class OpenGuardModel(BaseModel):
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, system_prompt:str) -> None:
        super().__init__(system_prompt)
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def generate(self, inputs: list[str], outputs:list[str], max_new_tokens: int, logger: Logger | None = None) -> list[str]:
        chat = []
        for i in range(len(inputs)):
            if self.system_prompt != None:
                chat_list = [
                    {"role" : "system", "content" : self.system_prompt},
                    {"role" : "user", "content" : inputs[i]},
                    {"role" : "assistant", "content" : outputs[i]},
                ]
            else:
                chat_list = [
                    {"role" : "user", "content" : inputs[i]},
                    {"role" : "assistant", "content" : outputs[i]},
                ]
            chat.append(self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True))
        tokenized_inputs = self.tokenizer(chat, return_tensors="pt", padding=True).to(self.device)
        gen = self.model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens)
        gen = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for i in range(len(gen)):
            gen[i] = gen[i][len(chat[i]):]
            if logger != None:
                logger.info(f"Input: {chat[i]}")
                logger.info(f"Output: {gen[i]}")
        return gen
    
    def judge(self, inputs:list[str], outputs:list[str], logger: Logger | None = None) -> tuple[list[bool], list[str]]:
        gen = self.generate(inputs, outputs, 1, logger)
        passed = [True if g.lower()=="safe" else False for g in gen]
        return passed, gen

# TODO
class CloseGuardModel(CloseModel):
    def __init__(self, model, model_type, system_prompt:str, api_key:str) -> None:
        super().__init__(model, model_type, system_prompt)
    
    def judge(self, inputs:list[str], outputs:list[str]) -> tuple[torch.Tensor, list[str]]:
        ...

def load_llm(path:str, model_type:str, model_key="", guard_path="", guard_type="", guard_key="", proxies:dict[str,str]|None=None) -> tuple[OpenModel|CloseModel, OpenGuardModel|CloseGuardModel]:
    model_type = model_type.lower()
    guard_type = guard_type.lower()
    if model_type in OPEN_LLM_TYPE_LIST:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto", attn_implementation="eager").eval()
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        system_prompt = SYS_PROMPT[model_type]
        gen_model = OpenModel(model, tokenizer, system_prompt)
    elif model_type in CLOSE_LLM_TYPE_LIST:
        gen_model = CloseModel(path, model_type, SYS_PROMPT[model_type], model_key, proxies)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if guard_path != "":
        if guard_type in OPEN_LLM_TYPE_LIST:
            guard_model = AutoModelForCausalLM.from_pretrained(guard_path, torch_dtype=torch.float16, device_map="balanced", attn_implementation="eager").eval()
            guard_tokenizer = AutoTokenizer.from_pretrained(guard_path, padding_side="left")
            if guard_tokenizer.pad_token is None:
                guard_tokenizer.pad_token = guard_tokenizer.eos_token
            guard_system_prompt = SYS_PROMPT[guard_type]
            guard = OpenGuardModel(guard_model, guard_tokenizer, guard_system_prompt)
        elif guard_type in CLOSE_LLM_TYPE_LIST:
            # TODO
            raise ValueError(f"Close-source guard model is not supported yet.")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        guard = None
    
    return gen_model, guard

def load_model_and_tokenizer(model_name:str, model_path:str, logger:Optional[Logger]=None, device="cuda:0"):
    # use "eager" attention to avoid different results between batched generation and single generation, 
    # see https://github.com/huggingface/transformers/issues/32390 and other related issues. 
    # Other attn_implementations might cause different logit outputs. 
    # Though in most cases the difference is small, we use "eager" implementation 
    # to avoid the influence of this issue. If you want to improve the speed of generation, 
    # consider using other implementations like "sdpa". 
    # But be aware that this might introduce other issues.
    hf_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager")
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
