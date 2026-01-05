import sys
import os
import torch
import functools
import einops
import gc
import random
random.seed(42)
import json
import time
import pickle
import psutil
import atexit
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, Qwen3ForCausalLM, Qwen2ForCausalLM
from jaxtyping import Float, Int
from collections import defaultdict
from transformers import PreTrainedTokenizer, PreTrainedModel
from loguru._logger import Logger
from loguru import logger
from typing import Optional
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

torch.set_grad_enabled(False)

from utils.prompt import SYS_PROMPT
from server.client import QwenAnswerEvaluatorClient, get_server, QwenGuardEvaluatorClient
from config import LVSConfig
from model import OpenModel, load_model_and_tokenizer, model_name_dict
from glue_eval.glue_eval import GLUEEval

from threading import Lock
savefig_lock = Lock()
from argparse import ArgumentParser

torch._dynamo.config.recompile_limit = 1024  # Set a higher limit to avoid excessive recompilations

logging.set_verbosity(logging.ERROR)

def split_texts(texts: List[str], split_ratio: list[float] = [0.8, 0.1, 0.1]):
    """
    Split texts into training, validation, and test sets.

    Parameters
    ----------
    texts : List[str]
        List of texts to be split.
    split_ratio : list[float], optional
        Ratios for train, val, test sets. Will be normalized if sum != 1.0.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (train_texts, val_texts, test_texts)
    """
    random.shuffle(texts)
    split_ratio = [r / sum(split_ratio) for r in split_ratio]  # Normalize the split ratios
    train_end = int(len(texts) * split_ratio[0])
    val_end = train_end + int(len(texts) * split_ratio[1])
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]

    return train_texts, val_texts, test_texts

def eval_answer_rate(
    model: OpenModel,
    answer_eval_agent: QwenAnswerEvaluatorClient,
    queries: List[str],
    eval_type: Optional[str] = None,
    batch_size: int = 16,
    disable_tqdm: bool = True,
    ):
    """
    Evaluate the answer rate of the model on queries.

    Parameters
    ----------
    model : OpenModel
        The model to be evaluated.
    answer_eval_agent : QwenAnswerEvaluatorClient
        The answer evaluation agent.
    queries : List[str]
        List of queries to be evaluated.
    eval_type : Optional[str], optional
        Evaluation type.
    batch_size : int, optional
        Batch size for evaluation.
    disable_tqdm : bool, optional
        Disable tqdm progress bar.

    Returns
    -------
    Tuple[List[str], List[bool], List[float]]
        (generated texts, answer booleans, answer rates)
    """
    gen = []
    answer_bools = []
    answer_rates = []
    for q in tqdm(range(0, len(queries), batch_size), disable=disable_tqdm):
        batch = queries[q:q+batch_size]
        batch_gen = model.generate(batch, 128, do_sample=False)
        if eval_type is not None:
            answer_eval_agent.input_queue.put((batch, batch_gen, eval_type))
        else:
            answer_eval_agent.input_queue.put((batch, batch_gen))
        gen.extend(batch_gen)
    answer_eval_agent.input_queue.join()  # Wait for all requests to be processed
    start_time = time.perf_counter()
    while not answer_eval_agent.result_queue.empty():
        batch_result = answer_eval_agent.result_queue.get()
        if isinstance(batch_result, Exception):
            logger.error(f"Error in answer evaluation: {batch_result}")
            continue
        batch_answer_bools, batch_answer_rates = batch_result
        answer_bools.extend(batch_answer_bools)
        answer_rates.extend(batch_answer_rates)
        answer_eval_agent.result_queue.task_done()
    end_time = time.perf_counter()
    if end_time - start_time > 10:
        logger.warning(f"Answer evaluation took too long: {end_time - start_time} seconds")
    return gen, answer_bools, answer_rates

def split_by_answered_status(inputs, gens, answer_rates, threshold=0.5):
    """
    Split inputs, generated texts, and answer rates into answered/unanswered by threshold.

    Parameters
    ----------
    inputs : List[str]
        List of input texts.
    gens : List[str]
        List of generated texts.
    answer_rates : List[float]
        List of answer rates.
    threshold : float, optional
        Threshold for considering answered.

    Returns
    -------
    Tuple[Tuple[List, List, List], Tuple[List, List, List]]
        ((answered_inputs, answered_gens, answered_rates), (unanswered_inputs, unanswered_gens, unanswered_rates))
    """
    answered_inputs = []
    answered_gens = []
    answered_rates = []
    unanswered_inputs = []
    unanswered_gens = []
    unanswered_rates = []
    for inp, gen, rate in zip(inputs, gens, answer_rates):
        if rate > threshold:
            answered_inputs.append(inp)
            answered_gens.append(gen)
            answered_rates.append(rate)
        else:
            unanswered_inputs.append(inp)
            unanswered_gens.append(gen)
            unanswered_rates.append(rate)
    return (answered_inputs, answered_gens, answered_rates), (unanswered_inputs, unanswered_gens, unanswered_rates)

def format_texts(texts:list[str], system_prompt: Optional[str] = None) -> List[List[dict]]:
    """
    Format input texts for the model's chat template.

    Parameters
    ----------
    texts : list[str]
        List of input texts.
    system_prompt : Optional[str], optional
        System prompt to include.

    Returns
    -------
    List[List[dict]]
        List of formatted chat dictionaries.
    """
    if system_prompt:
        return [[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}] for text in texts]
    else:
        return [[{"role": "user", "content": text}] for text in texts]

def tokenize_instructions(tokenizer: PreTrainedTokenizer, instructions: List[List[dict]], add_generation_prompt=True) -> Int[Tensor, "batch seq"]:
    """
    Tokenize input instructions using the provided tokenizer.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer to use.
    instructions : List[List[dict]]
        Instructions to tokenize.
    add_generation_prompt : bool, optional
        Whether to add generation prompt.

    Returns
    -------
    Int[Tensor, "batch seq"]
        Tokenized input IDs tensor.
    """
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    ).input_ids

def batch_run(hooked_model: HookedTransformer, tokenizer: PreTrainedTokenizer, batch: List[List[dict]]):
    """
    Run a batch of inputs through the model and return the cache.

    Parameters
    ----------
    hooked_model : HookedTransformer
        Model to run inputs through.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use.
    batch : List[List[dict]]
        Formatted input instructions.

    Returns
    -------
    dict
        Cache of model activations.
    """
    batch_logits, batch_cache = hooked_model.run_with_cache(
        tokenize_instructions(tokenizer, batch, add_generation_prompt=config.add_generation_prompt),
        names_filter=lambda hook_name: 'resid' in hook_name or 'out' in hook_name,
        reset_hooks_end=True,
        pos_slice=slice(-config.select_token_num, None),
    )
    final_cache = defaultdict(list)
    for k, v in batch_cache.items():
        final_cache[k] = v[:, -config.select_token_num:, :].reshape(-1, v.size(-1)).clone()
    return final_cache

def get_cache(hooked_model: HookedTransformer, tokenizer: PreTrainedTokenizer, inputs: List[str], batch_size: int = 16, system_prompt: Optional[str] = None):
    """
    Get cache for inputs by running them through the model in batches.

    Parameters
    ----------
    hooked_model : HookedTransformer
        Model to run inputs through.
    tokenizer : PreTrainedTokenizer
        Tokenizer to use.
    inputs : List[str]
        Input texts to process.
    batch_size : int, optional
        Batch size.
    system_prompt : Optional[str], optional
        System prompt to include.

    Returns
    -------
    dict
        Cache of model activations for the inputs.
    """
    formatted_texts = format_texts(inputs, system_prompt)
    cache = defaultdict(list)
    for q in tqdm(range(0, len(formatted_texts), batch_size), disable=True):
        batch = formatted_texts[q:q + batch_size]
        batch_cache = batch_run(
            hooked_model,
            tokenizer,
            batch
        )
        for k, v in batch_cache.items():
            cache[k].append(v)
        del batch_cache, batch
        gc.collect()
        torch.cuda.empty_cache()
    return {k: torch.cat(v, dim=0) for k, v in cache.items()}

def get_act_idx(cache_dict: dict[str, torch.Tensor], act_name: str, layer: int):
    """
    Get activation index from cache dictionary by activation name and layer.

    Parameters
    ----------
    cache_dict : dict[str, torch.Tensor]
        Cache dictionary.
    act_name : str
        Activation name.
    layer : int
        Layer number.

    Returns
    -------
    torch.Tensor or None
        Activation tensor for the specified name and layer.
    """
    key = (act_name, layer)
    return cache_dict.get(utils.get_act_name(*key), None)

def calculate_activation_and_std(layer:str, layer_num:int, plot:bool=False):
    toxic_answered_cache = get_act_idx(sampled_datasets["toxic_train"]["answered"]["cache"], layer, layer_num)
    toxic_unanswered_cache = get_act_idx(sampled_datasets["toxic_train"]["unanswered"]["cache"], layer, layer_num)
    benign_answered_cache = get_act_idx(sampled_datasets["benign_train"]["answered"]["cache"], layer, layer_num)
    benign_unanswered_cache = get_act_idx(sampled_datasets["benign_train"]["unanswered"]["cache"], layer, layer_num)

    # Some caches may be None. Replace None with an empty tensor with first dim 0
    caches = {
        "toxic_answered_cache": toxic_answered_cache,
        "toxic_unanswered_cache": toxic_unanswered_cache,
        "benign_answered_cache": benign_answered_cache,
        "benign_unanswered_cache": benign_unanswered_cache,
    }
    # Find a reference tensor to copy device/dtype/feature-dim from
    ref = next((t for t in caches.values() if t is not None), None)
    if ref is None:
        # No reference available; infer feature dim from model cfg if possible
        d_model = getattr(hooked_model.cfg, "d_model", None)
        if d_model is None:
            # fallback to 0-feature tensor (will likely break later, but avoids None)
            for k in caches:
                caches[k] = torch.empty((0, 0))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for k in caches:
                caches[k] = torch.empty((0, d_model), device=device, dtype=torch.float32)
    else:
        feat_dim = ref.size(-1)
        for k, v in list(caches.items()):
            if v is None:
                caches[k] = torch.empty((0, feat_dim), device=ref.device, dtype=ref.dtype)

    toxic_answered_cache = caches["toxic_answered_cache"]
    toxic_unanswered_cache = caches["toxic_unanswered_cache"]
    benign_answered_cache = caches["benign_answered_cache"]
    benign_unanswered_cache = caches["benign_unanswered_cache"]
    
    toxic_answered_val_cache = get_act_idx(sampled_datasets["toxic_val"]["answered"]["cache"], layer, layer_num)
    toxic_unanswered_val_cache = get_act_idx(sampled_datasets["toxic_val"]["unanswered"]["cache"], layer, layer_num)
    benign_answered_val_cache = get_act_idx(sampled_datasets["benign_val"]["answered"]["cache"], layer, layer_num)
    benign_unanswered_val_cache = get_act_idx(sampled_datasets["benign_val"]["unanswered"]["cache"], layer, layer_num)
    
    # Some caches may be None. Replace None with an empty tensor with first dim 0
    caches = {
        "toxic_answered_val_cache": toxic_answered_val_cache,
        "toxic_unanswered_val_cache": toxic_unanswered_val_cache,
        "benign_answered_val_cache": benign_answered_val_cache,
        "benign_unanswered_val_cache": benign_unanswered_val_cache,
    }
    # Find a reference tensor to copy device/dtype/feature-dim from
    ref = next((t for t in caches.values() if t is not None), None)
    if ref is None:
        # No reference available; infer feature dim from model cfg if possible
        d_model = getattr(hooked_model.cfg, "d_model", None)
        if d_model is None:
            # fallback to 0-feature tensor (will likely break later, but avoids None)
            for k in caches:
                caches[k] = torch.empty((0, 0))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for k in caches:
                caches[k] = torch.empty((0, d_model), device=device, dtype=torch.float32)
    else:
        feat_dim = ref.size(-1)
        for k, v in list(caches.items()):
            if v is None:
                caches[k] = torch.empty((0, feat_dim), device=ref.device, dtype=ref.dtype)

    toxic_answered_val_cache = caches["toxic_answered_val_cache"]
    toxic_unanswered_val_cache = caches["toxic_unanswered_val_cache"]
    benign_answered_val_cache = caches["benign_answered_val_cache"]
    benign_unanswered_val_cache = caches["benign_unanswered_val_cache"]
    
    all_activation = torch.cat([toxic_answered_cache, toxic_unanswered_cache, benign_answered_cache, benign_unanswered_cache], dim=0)
    
    val_activation = torch.cat([toxic_answered_val_cache, toxic_unanswered_val_cache, benign_answered_val_cache, benign_unanswered_val_cache], dim=0)
    
    svm = LinearSVC(fit_intercept=False, max_iter=2000)
    labels = np.concatenate([
        np.zeros(toxic_answered_cache.shape[0]), 
        np.zeros(toxic_unanswered_cache.shape[0]),
        np.ones(benign_answered_cache.shape[0]),
        np.ones(benign_unanswered_cache.shape[0])
    ]).astype(int)
    val_labels = np.concatenate([
        np.zeros(toxic_answered_val_cache.shape[0]), 
        np.zeros(toxic_unanswered_val_cache.shape[0]),
        np.ones(benign_answered_val_cache.shape[0]),
        np.ones(benign_unanswered_val_cache.shape[0])
    ]).astype(int)
    svm.fit(all_activation.float().cpu().numpy(), labels)
    # Calculate the accuracy of the SVM
    benign_dir_score_train = svm.score(all_activation.float().cpu().numpy(), labels)
    benign_dir_score = svm.score(val_activation.float().cpu().numpy(), val_labels)
    benign_dir = torch.tensor(svm.coef_[0], device=toxic_answered_cache.device, dtype=toxic_answered_cache.dtype)
    # randomly rotate the dir by config.degree degrees
    # Generate a random vector with the same shape as benign_dir
    random_vec = torch.randn_like(benign_dir)
    # Make it orthogonal to benign_dir
    random_vec = random_vec - benign_dir * (torch.dot(benign_dir, random_vec) / benign_dir.norm()**2)
    random_vec = random_vec / random_vec.norm()
    # Rotate benign_dir by config.degree degrees towards random_vec
    theta = torch.tensor(config.degree * np.pi / 180.0, device=benign_dir.device, dtype=benign_dir.dtype)
    benign_dir = benign_dir * torch.cos(theta) + random_vec * torch.sin(theta)
    benign_dir = benign_dir / benign_dir.norm(dim=-1)
    benign_std = all_activation.matmul(benign_dir).std()
    
    svm = LinearSVC(fit_intercept=False, max_iter=2000)
    labels = np.concatenate([
        np.ones(toxic_answered_cache.shape[0]), 
        np.zeros(toxic_unanswered_cache.shape[0]),
        np.ones(benign_answered_cache.shape[0]),
        np.zeros(benign_unanswered_cache.shape[0])
    ]).astype(int)
    val_labels = np.concatenate([
        np.ones(toxic_answered_val_cache.shape[0]), 
        np.zeros(toxic_unanswered_val_cache.shape[0]),
        np.ones(benign_answered_val_cache.shape[0]),
        np.zeros(benign_unanswered_val_cache.shape[0])
    ]).astype(int)
    svm.fit(all_activation.float().cpu().numpy(), labels)
    # Calculate the accuracy of the SVM
    answered_dir_score_train = svm.score(all_activation.float().cpu().numpy(), labels)
    answered_dir_score = svm.score(val_activation.float().cpu().numpy(), val_labels)
    answered_dir = torch.tensor(svm.coef_[0], device=toxic_answered_cache.device, dtype=toxic_answered_cache.dtype)
    # randomly rotate the dir by config.degree degrees
    # Generate a random vector with the same shape as answered_dir
    random_vec = torch.randn_like(answered_dir)
    # Make it orthogonal to answered_dir
    random_vec = random_vec - answered_dir * (torch.dot(answered_dir, random_vec) / answered_dir.norm()**2)
    random_vec = random_vec / random_vec.norm()
    # Rotate answered_dir by config.degree degrees towards random_vec
    theta = torch.tensor(config.degree * np.pi / 180.0, device=answered_dir.device, dtype=answered_dir.dtype)
    answered_dir = answered_dir * torch.cos(theta) + random_vec * torch.sin(theta)
    answered_dir = answered_dir / answered_dir.norm(dim=-1)
    answer_std = all_activation.matmul(answered_dir).std()
    
    # (Optional) Visualize the benign and answer vectors
    if plot and layer_num % 4 == 0:
        with savefig_lock:
            # Calculate the projections of the activations on the benign and answered directions
            x_toxic_answered = toxic_answered_cache.matmul(benign_dir).float().cpu().numpy()
            y_toxic_answered = toxic_answered_cache.matmul(answered_dir).float().cpu().numpy()
            x_toxic_unanswered = toxic_unanswered_cache.matmul(benign_dir).float().cpu().numpy()
            y_toxic_unanswered = toxic_unanswered_cache.matmul(answered_dir).float().cpu().numpy()
            x_benign_answered = benign_answered_cache.matmul(benign_dir).float().cpu().numpy()
            y_benign_answered = benign_answered_cache.matmul(answered_dir).float().cpu().numpy()
            x_benign_unanswered = benign_unanswered_cache.matmul(benign_dir).float().cpu().numpy()
            y_benign_unanswered = benign_unanswered_cache.matmul(answered_dir).float().cpu().numpy()

            plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
            main_ax = plt.subplot(grid[1:4, 0:3])
            y_hist = plt.subplot(grid[1:4, 3], sharey=main_ax)
            x_hist = plt.subplot(grid[0, 0:3], sharex=main_ax)

            # Scatter figure
            # Sample 100 points from each set
            def sample_points(x_data, y_data, n=100):
                if len(x_data) <= n:
                    return x_data, y_data
                indices = np.random.choice(len(x_data), n, replace=False)
                return x_data[indices], y_data[indices]

            # Sample 100 points from each dataset
            x_toxic_answered_sampled, y_toxic_answered_sampled = sample_points(x_toxic_answered, y_toxic_answered, 100)
            x_toxic_unanswered_sampled, y_toxic_unanswered_sampled = sample_points(x_toxic_unanswered, y_toxic_unanswered, 100)
            x_benign_answered_sampled, y_benign_answered_sampled = sample_points(x_benign_answered, y_benign_answered, 100)
            x_benign_unanswered_sampled, y_benign_unanswered_sampled = sample_points(x_benign_unanswered, y_benign_unanswered, 100)

            # Plot the sampled points
            main_ax.scatter(x_toxic_answered_sampled, y_toxic_answered_sampled, alpha=0.3, s=10, label="toxic_answered", color="red")
            main_ax.scatter(x_toxic_unanswered_sampled, y_toxic_unanswered_sampled, alpha=0.3, s=10, label="toxic_unanswered", color="orange")
            main_ax.scatter(x_benign_answered_sampled, y_benign_answered_sampled, alpha=0.3, s=10, label="benign_answered", color="blue")
            main_ax.scatter(x_benign_unanswered_sampled, y_benign_unanswered_sampled, alpha=0.3, s=10, label="benign_unanswered", color="green")
            main_ax.axvline(x=0, color='purple', linestyle='--', label='benign_split_point')
            main_ax.axhline(y=0, color='brown', linestyle='--', label='answer_split_point')
            main_ax.set_xlabel("Projection on benign_dir")
            main_ax.set_ylabel("Projection on answered_dir")
            main_ax.set_title(f"Layer {layer_num} {layer} projection scatter")
            main_ax.legend()

            # x axis distribution
            sns.kdeplot(x=np.concatenate([x_toxic_answered_sampled, x_toxic_unanswered_sampled, x_benign_answered_sampled, x_benign_unanswered_sampled]), ax=x_hist, fill=True)
            x_hist.axis("off")

            # y axis distribution
            sns.kdeplot(y=np.concatenate([y_toxic_answered_sampled, y_toxic_unanswered_sampled, y_benign_answered_sampled, y_benign_unanswered_sampled]), ax=y_hist, fill=True)
            y_hist.axis("off")

            os.makedirs(os.path.join(figure_path, "benign_and_answered_vectors", f"iter_{iteration}"), exist_ok=True)
            plt.savefig(os.path.join(figure_path, "benign_and_answered_vectors", f"iter_{iteration}", f"layer_{layer}_{layer_num}.pdf"), bbox_inches='tight')
            plt.close()
    return benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train

def vector_angle(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the angle in radians between two vectors.
    Parameters
    ----------
        v1 : torch.Tensor
            The first vector.
        v2 : torch.Tensor
            The second vector.
    Returns
    -------
        float
            The angle in radians between the two vectors.
    """
    v1_norm = v1 / v1.norm()
    v2_norm = v2 / v2.norm()
    cos_theta = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = torch.acos(cos_theta)
    return angle.item()

def get_modified_matrix(
    matrix: Float[Tensor, "... d_model"],
    benign_direction: Float[Tensor, "d_act"],
    benign_std: float,
    answer_direction: Float[Tensor, "d_act"],
    answer_std: float,
) -> Float[Tensor, "... d_model"]:
    """
    Modify the given matrix with respect to the benign and answer directions, scaling by their standard deviations.
    Parameters
    ----------
        matrix : Float[Tensor, "... d_model"]
            The matrix to be orthogonalized.
        benign_direction : Float[Tensor, "d_act"]
            The benign direction vector.
        benign_std : float
            The standard deviation for the benign direction.
        answer_direction : Float[Tensor, "d_act"]
            The answer direction vector.
        answer_std : float
            The standard deviation for the answer direction.
    Returns
    -------
        Float[Tensor, "... d_model"]
            The orthogonalized matrix.
    """
    benign_proj = (
        (einops.einsum(matrix, benign_direction.view(-1, 1), "... d_model, d_model single -> ... single"))/benign_std*answer_std * answer_direction
    )
    answer_proj = (
        (einops.einsum(matrix, answer_direction.view(-1, 1), "... d_model, d_model single -> ... single")) * answer_direction
    )
    return matrix - answer_proj + benign_proj

def load_modified_model(hooked_model: HookedTransformer, benign_direction_list: list[dict[str, list[list[float]]]], benign_std_list: list[dict[str, list[float]]], answer_direction_list: list[dict[str, list[list[float]]]], answer_std_list: list[dict[str, list[float]]], selected_layer_list: list[dict[str, list[int]]]) -> HookedTransformer:
    """
    Load a modified model (IN PLACE) with the specified benign and answer directions and their standard deviations.
    Parameters
    ----------
        hooked_model : HookedTransformer
            The original HookedTransformer model to be modified.
        benign_direction_list : list[dict[str, list[list[float]]]]
            List of benign direction vectors for each iteration and each layer.
        benign_std_list : list[dict[str, list[float]]]
            List of standard deviations for the benign directions for each iteration and each layer.
        answer_direction_list : list[dict[str, list[list[float]]]]
            List of answer direction vectors for each iteration and each layer.
        answer_std_list : list[dict[str, list[float]]]
            List of standard deviations for the answer directions for each iteration and each layer.
    Returns
    -------
        HookedTransformer
            The HookedTransformer model with the modified weights (IN PLACE).
    """
    for benign_activation, benign_std_dict, answer_activation, answer_std_dict, selected_layers in zip(benign_direction_list, benign_std_list, answer_direction_list, answer_std_list, selected_layer_list):
        for i, block in tqdm(enumerate(hooked_model.blocks), disable=True):
            if i in selected_layers["attn_out"]:
                block.attn.W_O.data[:] = get_modified_matrix(
                    block.attn.W_O,
                    benign_activation["attn_out"][i],
                    benign_std_dict["attn_out"][i],
                    answer_activation["attn_out"][i],
                    answer_std_dict["attn_out"][i]
                )
            if i in selected_layers["mlp_out"]:
                block.mlp.W_out.data[:] = get_modified_matrix(
                    block.mlp.W_out,
                    benign_activation["mlp_out"][i],
                    benign_std_dict["mlp_out"][i],
                    answer_activation["mlp_out"][i],
                    answer_std_dict["mlp_out"][i]
                )
    return hooked_model

def cleanup():
    # Kill any process running "src/server_answer.py"
    # !!! DO NOT USE THIS WHEN RUNNING SEVERAL INSTANCES OF THIS SCRIPT AT THE SAME TIME !!!
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any("src/server_answer.py" in arg for arg in cmdline):
                logger.info(f"Killing process {proc.pid} running: {' '.join(cmdline)}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
def sig_handler(signum, frame):
    cleanup()
    sys.exit(0)

def append_glue_results(glue_results, new_results):
    for ds, result_dict in new_results.items():
        if ds == "perplexity":
            glue_results[ds]["results"].append(result_dict)
            continue
        for metric, value in result_dict.items():
            glue_results[ds][metric].append(value)
    return glue_results

if __name__ == "__main__":
    # Load config
    # config = LVSConfig.get_config()
    parser = ArgumentParser()
    parser.add_argument("--degree", type=float, default=60.0, help="Degree to rotate the direction")
    parser = LVSConfig.add_args(parser)
    args = parser.parse_args()
    config = LVSConfig.get_config(vars(args))
    config.degree = args.degree
    
    # Init logger
    log_path = os.path.join(config.log_path, "loop_vector_steering", config.model_type)
    os.makedirs(log_path, exist_ok=True)
    logger.add(
        os.path.join(log_path, f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.log"),
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | process:{process} | {message}",
        level="DEBUG" if config.debug else "INFO",
    )
    logger.add(
        sys.stdout,
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | process:{process} | {message}",
        level="DEBUG" if config.debug else "INFO",
    )
    logger.remove(0)
    logger.info(f"Config: {config}")
    
    # Init result directory and config file
    result_path = os.path.join(config.result_path, config.model_type)
    os.makedirs(result_path, exist_ok=True)
    config.save_yaml(os.path.join(result_path, "config.yaml"))
    figure_path = os.path.join(config.figure_path, config.model_type)
    os.makedirs(figure_path, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading model {config.model_type} from {config.model_path}...")
    hf_model, hooked_model, tokenizer = load_model_and_tokenizer(model_name_dict[config.model_type], config.model_path, logger)
    system_prompt = SYS_PROMPT.get(config.model_type, None)
    open_model = OpenModel(hf_model, tokenizer, system_prompt)
    glue_eval = GLUEEval(model=hf_model, tokenizer=tokenizer, number_of_tests=256)
    
    split_dataset_path = "dataset/split_dataset"
    if not os.path.exists(split_dataset_path):
        os.makedirs(split_dataset_path)
    
    # Load dataset
    if os.path.exists(os.path.join(split_dataset_path, "seval_attack_train.csv")) and os.path.exists(os.path.join(split_dataset_path, "seval_attack_val.csv")) and os.path.exists(os.path.join(split_dataset_path, "seval_attack_test.csv")) and os.path.exists(os.path.join(split_dataset_path, "seval_risk_train.csv")) and os.path.exists(os.path.join(split_dataset_path, "seval_risk_val.csv")) and os.path.exists(os.path.join(split_dataset_path, "seval_risk_test.csv")) and os.path.exists(os.path.join(split_dataset_path, "orfuzzset_train.csv")) and os.path.exists(os.path.join(split_dataset_path, "orfuzzset_val.csv")) and os.path.exists(os.path.join(split_dataset_path, "orfuzzset_test.csv")) and os.path.exists(os.path.join(split_dataset_path, "nq_train.csv")) and os.path.exists(os.path.join(split_dataset_path, "nq_val.csv")) and os.path.exists(os.path.join(split_dataset_path, "nq_test.csv")):
        seval_attack_train = pd.read_csv(os.path.join(split_dataset_path, "seval_attack_train.csv"))["input"].tolist()
        seval_attack_val = pd.read_csv(os.path.join(split_dataset_path, "seval_attack_val.csv"))["input"].tolist()
        seval_attack_test = pd.read_csv(os.path.join(split_dataset_path, "seval_attack_test.csv"))["input"].tolist()
        seval_risk_train = pd.read_csv(os.path.join(split_dataset_path, "seval_risk_train.csv"))["input"].tolist()
        seval_risk_val = pd.read_csv(os.path.join(split_dataset_path, "seval_risk_val.csv"))["input"].tolist()
        seval_risk_test = pd.read_csv(os.path.join(split_dataset_path, "seval_risk_test.csv"))["input"].tolist()
        orfuzzset_train = pd.read_csv(os.path.join(split_dataset_path, "orfuzzset_train.csv"))["input"].tolist()
        orfuzzset_val = pd.read_csv(os.path.join(split_dataset_path, "orfuzzset_val.csv"))["input"].tolist()
        orfuzzset_test = pd.read_csv(os.path.join(split_dataset_path, "orfuzzset_test.csv"))["input"].tolist()
        nq_train = pd.read_csv(os.path.join(split_dataset_path, "nq_train.csv"))["input"].tolist()
        nq_val = pd.read_csv(os.path.join(split_dataset_path, "nq_val.csv"))["input"].tolist()
        nq_test = pd.read_csv(os.path.join(split_dataset_path, "nq_test.csv"))["input"].tolist()
    else:
        seval_attack = pd.read_csv("/your_data_path/result/gen_and_eval/seval_attack_sorted_result.csv")["input"].tolist()
        seval_risk = pd.read_csv("/your_data_path/result/gen_and_eval/seval_risk_sorted_result.csv")["input"].tolist()
        orfuzzset = pd.read_csv("/your_data_path/result/gen_and_eval/orfuzzset_sorted_result.csv")["input"].tolist()
        nq_dataset = pd.read_csv("/your_data_path/result/gen_and_eval/nq_sorted_result.csv")["input"].tolist()
        
        seval_attack = seval_attack[:500]
        seval_attack_train, seval_attack_val, seval_attack_test = split_texts(seval_attack)
        seval_risk = seval_risk[:500]
        seval_risk_train, seval_risk_val, seval_risk_test = split_texts(seval_risk)
        orfuzzset = orfuzzset[:500]
        orfuzzset_train, orfuzzset_val, orfuzzset_test = split_texts(orfuzzset)
        nq_dataset = nq_dataset[:500]
        nq_train, nq_val, nq_test = split_texts(nq_dataset)
        
        pd.DataFrame({"input": seval_attack_train}).to_csv(os.path.join(split_dataset_path, "seval_attack_train.csv"), index=False)
        pd.DataFrame({"input": seval_attack_val}).to_csv(os.path.join(split_dataset_path, "seval_attack_val.csv"), index=False)
        pd.DataFrame({"input": seval_attack_test}).to_csv(os.path.join(split_dataset_path, "seval_attack_test.csv"), index=False)
        pd.DataFrame({"input": seval_risk_train}).to_csv(os.path.join(split_dataset_path, "seval_risk_train.csv"), index=False)
        pd.DataFrame({"input": seval_risk_val}).to_csv(os.path.join(split_dataset_path, "seval_risk_val.csv"), index=False)
        pd.DataFrame({"input": seval_risk_test}).to_csv(os.path.join(split_dataset_path, "seval_risk_test.csv"), index=False)
        pd.DataFrame({"input": orfuzzset_train}).to_csv(os.path.join(split_dataset_path, "orfuzzset_train.csv"), index=False)
        pd.DataFrame({"input": orfuzzset_val}).to_csv(os.path.join(split_dataset_path, "orfuzzset_val.csv"), index=False)
        pd.DataFrame({"input": orfuzzset_test}).to_csv(os.path.join(split_dataset_path, "orfuzzset_test.csv"), index=False)
        pd.DataFrame({"input": nq_train}).to_csv(os.path.join(split_dataset_path, "nq_train.csv"), index=False)
        pd.DataFrame({"input": nq_val}).to_csv(os.path.join(split_dataset_path, "nq_val.csv"), index=False)
        pd.DataFrame({"input": nq_test}).to_csv(os.path.join(split_dataset_path, "nq_test.csv"), index=False)

    logger.info(f"Loaded {len(seval_attack_train)} seval attack training queries, {len(seval_attack_val)} seval attack validation queries, {len(seval_attack_test)} seval attack test queries, {len(seval_risk_train)} seval risk training queries, {len(seval_risk_val)} seval risk validation queries, {len(seval_risk_test)} seval risk test queries, {len(orfuzzset_train)} orfuzzset training queries, {len(orfuzzset_val)} orfuzzset validation queries, {len(orfuzzset_test)} orfuzzset test queries, {len(nq_train)} nq training queries, {len(nq_val)} nq validation queries, {len(nq_test)} nq test queries.")
    
    toxic_train_dict = {
        "seval_attack": seval_attack_train,
        "seval_risk": seval_risk_train,
    }
    toxic_val_dict = {
        "seval_attack": seval_attack_val,
        "seval_risk": seval_risk_val,
    }
    toxic_test_dict = {
        "seval_attack": seval_attack_test,
        "seval_risk": seval_risk_test,
    }
    benign_train_dict = {
        "orfuzzset": orfuzzset_train,
        "nq": nq_train,
    }
    benign_val_dict = {
        "orfuzzset": orfuzzset_val,
        "nq": nq_val,
    }
    benign_test_dict = {
        "orfuzzset": orfuzzset_test,
        "nq": nq_test,
    }
    
    # Load answer evaluator
    qwen_guard_eval_agent = QwenGuardEvaluatorClient(get_server(config.server_url, config.server_port))
    
    if config.kill_server:
        # Register cleanup function to be called on exit
        # This will ensure that the server is properly shut down
        atexit.register(cleanup)
        signal.signal(signal.SIGTERM, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)
    
    # Metrics to save or visualize
    f1_results = []
    seval_attack_results = []
    seval_risk_results = []
    nq_results = []
    orfuzzset_results = []
    glue_results = {
        "sst": defaultdict(list),
        "mrpc": defaultdict(list),
        "cola": defaultdict(list),
        "rte": defaultdict(list),
        "mmlu": defaultdict(list),
        "sentiment_analysis": defaultdict(list),
        "nli": defaultdict(list),
        "dialogue": defaultdict(list),
        "perplexity": defaultdict(list),
    }
    benign_direction_list: list[dict[str, list[list[float]]]] = [] # [iterations, layer_name, layer_num, d_model]
    benign_std_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    benign_score_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    benign_score_train_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    answer_direction_list: list[dict[str, list[list[float]]]] = [] # [iterations, layer_name, layer_num, d_model]
    answer_std_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    answer_score_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    answer_score_train_list: list[dict[str, list[float]]] = [] # [iterations, layer_name, layer_num]
    selected_layers_list: list[dict[str, list[int]]] = []  # [iterations, layer_name, layer_num]
    
    # Continue if there are existing results
    iter_path = [d for d in os.listdir(result_path) if d.startswith("iter_") and os.path.isdir(os.path.join(result_path, d))]
    if iter_path:
        max_iter = max([int(d.split("_")[1]) for d in iter_path])
        max_iter_path = os.path.join(result_path, f"iter_{max_iter}")
        logger.info(f"Found existing results, continuing from the last iteration {max_iter}...")
        begin_iter = max_iter + 1
        # Load the results from the last iteration
        result_df = pd.read_csv(os.path.join(max_iter_path, "results.csv"))
        f1_results = result_df["f1"].tolist()
        seval_attack_results = result_df["seval_attack"].tolist()
        seval_risk_results = result_df["seval_risk"].tolist()
        nq_results = result_df["nq"].tolist()
        orfuzzset_results = result_df["orfuzzset"].tolist()
        with open(os.path.join(max_iter_path, "benign_direction.pkl"), "rb") as f:
            benign_direction_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "benign_std.pkl"), "rb") as f:
            benign_std_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "benign_score.pkl"), "rb") as f:
            benign_score_list = pickle.load(f)
        if os.path.exists(os.path.join(max_iter_path, "benign_score_train.pkl")):
            with open(os.path.join(max_iter_path, "benign_score_train.pkl"), "rb") as f:
                benign_score_train_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "answer_direction.pkl"), "rb") as f:
            answer_direction_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "answer_std.pkl"), "rb") as f:
            answer_std_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "answer_score.pkl"), "rb") as f:
            answer_score_list = pickle.load(f)
        if os.path.exists(os.path.join(max_iter_path, "answer_score_train.pkl")):
            with open(os.path.join(max_iter_path, "answer_score_train.pkl"), "rb") as f:
                answer_score_train_list = pickle.load(f)
        with open(os.path.join(max_iter_path, "glue_results.pkl"), "rb") as f:
            glue_results = pickle.load(f)
        with open(os.path.join(max_iter_path, "selected_layers_list.pkl"), "rb") as f:
            selected_layers_list = pickle.load(f)
        hooked_model = load_modified_model(
            hooked_model,
            benign_direction_list,
            benign_std_list,
            answer_direction_list,
            answer_std_list,
            selected_layers_list,
        )
        logger.info(f"Loaded results from iteration {max_iter}.")
    else:
        logger.info("No existing results found, starting from scratch.")
        begin_iter = 0
    # Before starting the loop, we need to get the initial generations and answer rates
    logger.info("Evaluating the original model...")
    toxic_train_sampled_dict = {k: (random.sample(v, config.n_samples // len(toxic_train_dict)) if len(v) > config.n_samples // len(toxic_train_dict) else v) for k, v in toxic_train_dict.items()}
    benign_train_sampled_dict = {k: (random.sample(v, config.n_samples // len(benign_train_dict)) if len(v) > config.n_samples // len(benign_train_dict) else v) for k, v in benign_train_dict.items()}
    # Reorder the datasets by length to optimize the evaluation speed
    toxic_train_sampled_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in toxic_train_sampled_dict.items()}
    benign_train_sampled_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in benign_train_sampled_dict.items()}
    toxic_val_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in toxic_val_dict.items()}
    benign_val_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in benign_val_dict.items()}
    logger.info("Evaluating answer rate of toxic training set...")
    start_time = time.perf_counter()
    toxic_train_gen, toxic_train_answer_bools, toxic_train_answer_rates = {}, {}, {}
    for dataset_name, toxic_train in toxic_train_sampled_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=toxic_train,
            eval_type="jb",
            batch_size=config.batch_size,
        )
        toxic_train_gen[dataset_name] = g
        toxic_train_answer_bools[dataset_name] = b
        toxic_train_answer_rates[dataset_name] = r
    end_time = time.perf_counter()
    logger.info(f"Time taken to evaluate {sum(len(v) for v in toxic_train_sampled_dict.values())} toxic training samples: {end_time - start_time:.2f} seconds.")
    logger.info("Evaluating answer rate of benign training set...")
    start_time = time.perf_counter()
    benign_train_gen, benign_train_answer_bools, benign_train_answer_rates = {}, {}, {}
    for dataset_name, benign_train in benign_train_sampled_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=benign_train,
            eval_type="or",
            batch_size=config.batch_size,
        )
        benign_train_gen[dataset_name] = g
        benign_train_answer_bools[dataset_name] = b
        benign_train_answer_rates[dataset_name] = r
    end_time = time.perf_counter()
    logger.info(f"Time taken to evaluate {sum(len(v) for v in benign_train_sampled_dict.values())} benign training samples: {end_time - start_time:.2f} seconds.")
    # Evaluate the answer rate of the validation set and print the results
    logger.info("Evaluating answer rate of toxic validation set...")
    start_time = time.perf_counter()
    answer_labels = []
    answer_results = []
    toxic_val_gen, toxic_val_answer_bools, toxic_val_answer_rates = {}, {}, {}
    for dataset_name, toxic_val in toxic_val_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=toxic_val,
            eval_type="jb",
            batch_size=config.batch_size,
        )
        toxic_val_gen[dataset_name] = g
        toxic_val_answer_bools[dataset_name] = b
        toxic_val_answer_rates[dataset_name] = r
        answer_labels.extend(len(b)*[False])
        answer_results.extend(b)
    end_time = time.perf_counter()
    logger.info(f"Time taken to evaluate {sum(len(v) for v in toxic_val_dict.values())} toxic validation samples: {end_time - start_time:.2f} seconds.")
    logger.info("Evaluating answer rate of benign validation set...")
    start_time = time.perf_counter()
    benign_val_gen, benign_val_answer_bools, benign_val_answer_rates = {}, {}, {}
    for dataset_name, benign_val in benign_val_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=benign_val,
            eval_type="or",
            batch_size=config.batch_size,
        )
        benign_val_gen[dataset_name] = g
        benign_val_answer_bools[dataset_name] = b
        benign_val_answer_rates[dataset_name] = r
        answer_labels.extend(len(b)*[True])
        answer_results.extend(b)
    end_time = time.perf_counter()
    logger.info(f"Time taken to evaluate {sum(len(v) for v in benign_val_dict.values())} benign validation samples: {end_time - start_time:.2f} seconds.")
    # Evaluate the GLUE benchmark
    logger.info("Evaluating GLUE benchmark...")
    start_time = time.perf_counter()
    glue_result = glue_eval.evaluate(perplexity_flag=True, sst_flag=True, mmlu_flag=True, mrpc_flag=True, cola_flag=True, rte_flag=True, sentiment_analysis_flag=True, nli_flag=True, dialogue_flag=True, gen_len=5, batch_size=config.batch_size)
    end_time = time.perf_counter()
    logger.info(f"Time taken to evaluate GLUE benchmark: {end_time - start_time:.2f} seconds.")
    
    if begin_iter == 0:
        seval_attack_results.append(sum(toxic_val_answer_bools["seval_attack"]) / len(toxic_val_answer_bools["seval_attack"]))
        logger.info(f"seval_attack_results: {seval_attack_results[-1]}")
        seval_risk_results.append(sum(toxic_val_answer_bools["seval_risk"]) / len(toxic_val_answer_bools["seval_risk"]))
        logger.info(f"seval_risk_results: {seval_risk_results[-1]}")
        orfuzzset_results.append(1 - sum(benign_val_answer_bools["orfuzzset"]) / len(benign_val_answer_bools["orfuzzset"]))
        logger.info(f"orfuzzset_results: {orfuzzset_results[-1]}")
        nq_results.append(1 - sum(benign_val_answer_bools["nq"]) / len(benign_val_answer_bools["nq"]))
        logger.info(f"nq_results: {nq_results[-1]}")

        f1_results.append(f1_score(answer_labels, answer_results))
        logger.info("Final Evaluation Results:")
        logger.info(f"F1 Score: {f1_results[-1]}")
        glue_results = append_glue_results(glue_results, glue_result)
        logger.info(f"GLUE sst f1 score: {glue_result['sst']['f1']}")
        logger.info(f"GLUE sst MCC score: {glue_result['sst']['mcc']}")
        
        # Evaluate the initial model on the test set
        logger.info("Evaluating the initial model on the test set...")
        answer_labels = []
        answer_results = []
        toxic_test_gen, toxic_test_answer_bools, toxic_test_answer_rates = {}, {}, {}
        for dataset_name, toxic_test in toxic_test_dict.items():
            g, b, r = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=toxic_test,
                eval_type="jb",
                batch_size=config.batch_size,
            )
            toxic_test_gen[dataset_name] = g
            toxic_test_answer_bools[dataset_name] = b
            toxic_test_answer_rates[dataset_name] = r
            answer_labels.extend(len(b)*[False])
            answer_results.extend(b)
            logger.info(f"ASR for {dataset_name}: {sum(b) / len(b)}")
        benign_test_gen, benign_test_answer_bools, benign_test_answer_rates = {}, {}, {}
        for dataset_name, benign_test in benign_test_dict.items():
            g, b, r = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=benign_test,
                eval_type="or",
                batch_size=config.batch_size,
            )
            benign_test_gen[dataset_name] = g
            benign_test_answer_bools[dataset_name] = b
            benign_test_answer_rates[dataset_name] = r
            answer_labels.extend(len(b)*[True])
            answer_results.extend(b)
            logger.info(f"ORR for {dataset_name}: {1 - sum(b) / len(b)}")
                
        f1_test_result = f1_score(answer_labels, answer_results)
        logger.info("Final Evaluation Results on Test Set:")
        logger.info(f"F1 Test Result: {f1_test_result}")
        logger.info(f"GLUE sst f1 score: {glue_result['sst']['f1']}")
        logger.info(f"GLUE sst MCC score: {glue_result['sst']['mcc']}")
        
        # Save the test results
        with open(os.path.join(result_path, "test_original_results.json"), "w") as f:
            json.dump({
                "f1": f1_test_result,
                "seval_attack": sum(toxic_test_answer_bools["seval_attack"]) / len(toxic_test_answer_bools["seval_attack"]),
                "seval_risk": sum(toxic_test_answer_bools["seval_risk"]) / len(toxic_test_answer_bools["seval_risk"]),
                "orfuzzset": 1 - sum(benign_test_answer_bools["orfuzzset"]) / len(benign_test_answer_bools["orfuzzset"]),
                "nq": 1 - sum(benign_test_answer_bools["nq"]) / len(benign_test_answer_bools["nq"]),
                "glue": glue_result
            }, f, indent=4)
    else:
        for dataset_name, toxic_val in toxic_val_dict.items():
            logger.info(f"ASR for {dataset_name}: {sum(toxic_val_answer_bools[dataset_name]) / len(toxic_val_answer_bools[dataset_name])}")
        for dataset_name, benign_val in benign_val_dict.items():
            logger.info(f"ORR for {dataset_name}: {1 - sum(benign_val_answer_bools[dataset_name]) / len(benign_val_answer_bools[dataset_name])}")
        logger.info("Final Evaluation Results:")
        logger.info(f"F1 Score: {f1_score(answer_labels, answer_results)}")
        logger.info(f"GLUE sst f1 score: {glue_result['sst']['f1']}")
        logger.info(f"GLUE sst MCC score: {glue_result['sst']['mcc']}")

    # Iterative modification
    for iteration in range(begin_iter, config.max_iter):
        logger.info(f"Iteration {iteration}/{config.max_iter-1}")

        # Split the training set into answered and unanswered queries
        datasets = {
            "toxic_train": {
                "inputs": sum([toxic_train for toxic_train in toxic_train_sampled_dict.values()], start=[]),
                "gens": sum([gen for gen in toxic_train_gen.values()], start=[]),
                "rates": sum([rates for rates in toxic_train_answer_rates.values()], start=[])
            },
            "benign_train": {
                "inputs": sum([benign_train for benign_train in benign_train_sampled_dict.values()], start=[]),
                "gens": sum([gen for gen in benign_train_gen.values()], start=[]),
                "rates": sum([rates for rates in benign_train_answer_rates.values()], start=[])
            },
            "toxic_val": {
                "inputs": sum([toxic_val for toxic_val in toxic_val_dict.values()], start=[]),
                "gens": sum([gen for gen in toxic_val_gen.values()], start=[]),
                "rates": sum([rates for rates in toxic_val_answer_rates.values()], start=[])
            },
            "benign_val": {
                "inputs": sum([benign_val for benign_val in benign_val_dict.values()], start=[]),
                "gens": sum([gen for gen in benign_val_gen.values()], start=[]),
                "rates": sum([rates for rates in benign_val_answer_rates.values()], start=[])
            }
        }
        datasets_answer_split = {}
        for name, data in datasets.items():
            (answered_inputs, answered_gens, answered_rates), (unanswered_inputs, unanswered_gens, unanswered_rates) = split_by_answered_status(data["inputs"], data["gens"], data["rates"])
            datasets_answer_split[name] = {
                "answered": {
                    "inputs": answered_inputs,
                    "gens": answered_gens,
                    "rates": answered_rates
                },
                "unanswered": {
                    "inputs": unanswered_inputs,
                    "gens": unanswered_gens,
                    "rates": unanswered_rates
                }
            }
        # Sample data here if the dataset is too large
        sampled_datasets = datasets_answer_split
        
        # Get the cache for the answered and unanswered datasets
        logger.info("Getting cache...")
        start_time = time.perf_counter()
        error = False
        for name in sampled_datasets:
            for status in ["answered", "unanswered"]:
                inputs = sampled_datasets[name][status]["inputs"]
                logger.info(f"Processing {name} {status} dataset with {len(inputs)} inputs")
                cache = get_cache(hooked_model, tokenizer, inputs, batch_size=config.batch_size, system_prompt=SYS_PROMPT.get(config.model_type, None))
                sampled_datasets[name][status]["cache"] = cache
        end_time = time.perf_counter()
        logger.info(f"Time taken to get cache: {end_time - start_time:.2f} seconds.")

        # Calculate the benign vector and answer vector for each layer
        logger.info("Calculating benign and answer vectors...")
        activation_layers = ["attn_out", "mlp_out"]
        benign_activation = defaultdict(list)
        benign_std_dict = defaultdict(list)
        benign_score_dict = defaultdict(list)
        benign_score_train_dict = defaultdict(list)

        answer_activation = defaultdict(list)
        answer_std_dict = defaultdict(list)
        answer_score_dict = defaultdict(list)
        answer_score_train_dict = defaultdict(list)

        import concurrent.futures

        def calc_layer(layer_num, layer):
            benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train = calculate_activation_and_std(layer, layer_num, plot=True)
            return (layer, layer_num, benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train)

        # Prepare all (layer_num, layer) pairs in order
        tasks = [(layer_num, layer) for layer_num in range(hooked_model.cfg.n_layers) for layer in activation_layers]

        # Store results in a dict: {layer: [results by layer_num order]}
        results_dict = {layer: [] for layer in activation_layers}
        benign_std_dict = {layer: [] for layer in activation_layers}
        benign_score_dict = {layer: [] for layer in activation_layers}
        benign_score_train_dict = {layer: [] for layer in activation_layers}
        answer_activation = {layer: [] for layer in activation_layers}
        answer_std_dict = {layer: [] for layer in activation_layers}
        answer_score_dict = {layer: [] for layer in activation_layers}
        answer_score_train_dict = {layer: [] for layer in activation_layers}
        benign_activation = {layer: [] for layer in activation_layers}

        # Use ThreadPoolExecutor for parallelism (ProcessPool may have issues with CUDA tensors)
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(16) as executor:
            future_to_task = {executor.submit(calc_layer, layer_num, layer): (layer_num, layer) for layer_num, layer in tasks}
            # Collect results in the order of tasks
            results = [None] * len(tasks)
            for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
                res = future.result()
                layer, layer_num, benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train = res
                # Find the correct index to insert (to preserve order)
                idx = layer_num * len(activation_layers) + activation_layers.index(layer)
                results[idx] = res

        # Now, fill the dicts/lists in order
        for res in results:
            layer, layer_num, benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train = res
            benign_activation[layer].append(benign_dir)
            benign_std_dict[layer].append(benign_std)
            benign_score_dict[layer].append(benign_dir_score)
            benign_score_train_dict[layer].append(benign_dir_score_train)
            answer_activation[layer].append(answered_dir)
            answer_std_dict[layer].append(answer_std)
            answer_score_dict[layer].append(answered_dir_score)
            answer_score_train_dict[layer].append(answered_dir_score_train)
        
        benign_direction_list.append(benign_activation)
        benign_std_list.append(benign_std_dict)
        benign_score_list.append(benign_score_dict)
        benign_score_train_list.append(benign_score_train_dict)
        answer_direction_list.append(answer_activation)
        answer_std_list.append(answer_std_dict)
        answer_score_list.append(answer_score_dict)
        answer_score_train_list.append(answer_score_train_dict)
        end_time = time.perf_counter()
        logger.info(f"Time taken to calculate benign and answer vectors: {end_time - start_time:.2f} seconds.")

        # Visualize the angles between benign and answer vectors
        plt.figure(figsize=(8, 4))
        plt.axhline(y=90, color='r', linestyle='--', label='90 degrees')
        for layer in activation_layers:
            angles = []
            for l in range(hooked_model.cfg.n_layers):
                angle = vector_angle(
                    benign_activation[layer][l],
                    answer_activation[layer][l]
                )
                angles.append(angle * 180 / 3.1415926)

            plt.plot(range(hooked_model.cfg.n_layers), angles, marker='o', label=layer)
            plt.ylim(0, 120)
            plt.xlabel('Layer')
            plt.ylabel('Angle (degrees)')
            plt.title('Angle between benign and answered activation vs. Layer')
            plt.legend()
            plt.grid(True)
        os.makedirs(os.path.join(figure_path, "angle_between_benign_and_answered_activation"), exist_ok=True)
        plt.savefig(os.path.join(figure_path, "angle_between_benign_and_answered_activation", f"iter_{iteration}.pdf"), bbox_inches='tight')
        plt.close()
        
        # Visualize the scores
        plt.figure(figsize=(8, 4))
        for layer in activation_layers:
            plt.plot(range(hooked_model.cfg.n_layers), benign_score_dict[layer], marker='o', label=f"{layer} benign score test")
            plt.plot(range(hooked_model.cfg.n_layers), answer_score_dict[layer], marker='o', label=f"{layer} answered score test")
            plt.plot(range(hooked_model.cfg.n_layers), benign_score_train_dict[layer], marker='x', label=f"{layer} benign score train")
            plt.plot(range(hooked_model.cfg.n_layers), answer_score_train_dict[layer], marker='x', label=f"{layer} answered score train")
        plt.xlabel('Layer')
        plt.ylabel('Score')
        plt.title('Score of benign and answered activation vs. Layer')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(figure_path, "benign_and_answered_scores"), exist_ok=True)
        plt.savefig(os.path.join(figure_path, "benign_and_answered_scores", f"iter_{iteration}.pdf"), bbox_inches='tight')
        plt.close()

        # Calculate the contribution of each answer_activation to the final residual
        start_time = time.perf_counter()
        benign_dir, benign_std, benign_dir_score, benign_dir_score_train, answered_dir, answer_std, answered_dir_score, answered_dir_score_train = calculate_activation_and_std("resid_post", hooked_model.cfg.n_layers - 1, plot=False)
        benign_contribution = defaultdict(list)
        for layer in activation_layers:
            for l in range(hooked_model.cfg.n_layers):
                benign_contribution[layer].append(benign_activation[layer][l].matmul(benign_dir).float().cpu().item()*benign_score_dict[layer][l])

        answer_contribution = defaultdict(list)
        for layer in activation_layers:
            for l in range(hooked_model.cfg.n_layers):
                answer_contribution[layer].append(answer_activation[layer][l].matmul(answered_dir).float().cpu().item()*answer_score_dict[layer][l])

        best_layers = np.argsort(sum([(np.array(answer_contribution[layer])+np.array(benign_contribution[layer])).tolist() for layer in activation_layers], start=[]))[-config.select_layer_num:]
        selected_layers:dict[str, list[int]] = defaultdict(list)
        for l in best_layers:
            selected_layers[activation_layers[l // hooked_model.cfg.n_layers]].append(l % hooked_model.cfg.n_layers)
        logger.info(f"Selected layers: {selected_layers}")
        selected_layers_list.append(selected_layers)
        end_time = time.perf_counter()
        logger.info(f"Time taken to select layers: {end_time - start_time:.2f} seconds.")

        for i, block in tqdm(enumerate(hooked_model.blocks), disable=True):
            if i in selected_layers["attn_out"]:
                block.attn.W_O.data[:] = get_modified_matrix(block.attn.W_O, benign_activation["attn_out"][i], benign_std_dict["attn_out"][i], answer_activation["attn_out"][i], answer_std_dict["attn_out"][i])
            if i in selected_layers["mlp_out"]:
                block.mlp.W_out.data[:] = get_modified_matrix(block.mlp.W_out, benign_activation["mlp_out"][i], benign_std_dict["mlp_out"][i], answer_activation["mlp_out"][i], answer_std_dict["mlp_out"][i])

        # Evaluate the answer rate of the training set
        toxic_train_sampled_dict = {k: (random.sample(v, config.n_samples // len(toxic_train_dict)) if len(v) > config.n_samples // len(toxic_train_dict) else v) for k, v in toxic_train_dict.items()}
        benign_train_sampled_dict = {k: (random.sample(v, config.n_samples // len(benign_train_dict)) if len(v) > config.n_samples // len(benign_train_dict) else v) for k, v in benign_train_dict.items()}
        # Reorder the datasets by length to optimize the evaluation speed
        toxic_train_sampled_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in toxic_train_sampled_dict.items()}
        benign_train_sampled_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in benign_train_sampled_dict.items()}
        toxic_val_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in toxic_val_dict.items()}
        benign_val_dict = {k: sorted(v, key=lambda x: len(x), reverse=True) for k, v in benign_val_dict.items()}
        logger.info("Evaluating answer rate of toxic training set...")
        start_time = time.perf_counter()
        toxic_train_gen, toxic_train_answer_bools, toxic_train_answer_rates = {}, {}, {}
        for dataset_name, toxic_train in toxic_train_sampled_dict.items():
            toxic_train_gen[dataset_name], toxic_train_answer_bools[dataset_name], toxic_train_answer_rates[dataset_name] = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=toxic_train,
                eval_type="jb",
                batch_size=config.batch_size,
            )
        end_time = time.perf_counter()
        logger.info(f"Time taken to evaluate {sum(len(v) for v in toxic_train_sampled_dict.values())} toxic training samples: {end_time - start_time:.2f} seconds.")
        logger.info("Evaluating answer rate of benign training set...")
        start_time = time.perf_counter()
        benign_train_gen, benign_train_answer_bools, benign_train_answer_rates = {}, {}, {}
        for dataset_name, benign_train in benign_train_sampled_dict.items():
            benign_train_gen[dataset_name], benign_train_answer_bools[dataset_name], benign_train_answer_rates[dataset_name] = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=benign_train,
                eval_type="or",
                batch_size=config.batch_size,
            )
        end_time = time.perf_counter()
        logger.info(f"Time taken to evaluate {sum(len(v) for v in benign_train_sampled_dict.values())} benign training samples: {end_time - start_time:.2f} seconds.")
        # Evaluate the answer rate of the validation set and output the results
        logger.info("Evaluating answer rate of toxic validation set...")
        start_time = time.perf_counter()
        answer_labels = []
        answer_results = []
        toxic_val_gen, toxic_val_answer_bools, toxic_val_answer_rates = {}, {}, {}
        for dataset_name, toxic_val in toxic_val_dict.items():
            g, b, r = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=toxic_val,
                eval_type="jb",
                batch_size=config.batch_size,
            )
            toxic_val_gen[dataset_name] = g
            toxic_val_answer_bools[dataset_name] = b
            toxic_val_answer_rates[dataset_name] = r
            answer_labels.extend(len(b)*[False])
            answer_results.extend(b)
        end_time = time.perf_counter()
        logger.info(f"Time taken to evaluate {sum(len(v) for v in toxic_val_dict.values())} toxic validation samples: {end_time - start_time:.2f} seconds.")
        logger.info("Evaluating answer rate of benign validation set...")
        start_time = time.perf_counter()
        benign_val_gen, benign_val_answer_bools, benign_val_answer_rates = {}, {}, {}
        for dataset_name, benign_val in benign_val_dict.items():
            g, b, r = eval_answer_rate(
                model=open_model,
                answer_eval_agent=qwen_guard_eval_agent,
                queries=benign_val,
                eval_type="or",
                batch_size=config.batch_size,
            )
            benign_val_gen[dataset_name] = g
            benign_val_answer_bools[dataset_name] = b
            benign_val_answer_rates[dataset_name] = r
            answer_labels.extend(len(b)*[True])
            answer_results.extend(b)
        end_time = time.perf_counter()
        logger.info(f"Time taken to evaluate {sum(len(v) for v in benign_val_dict.values())} benign validation samples: {end_time - start_time:.2f} seconds.")
        # Evaluate the GLUE benchmark
        logger.info("Evaluating GLUE benchmark...")
        start_time = time.perf_counter()
        
        glue_result = glue_eval.evaluate(perplexity_flag=True, sst_flag=True, mmlu_flag=True, mrpc_flag=True, cola_flag=True, rte_flag=True, sentiment_analysis_flag=True, nli_flag=True, dialogue_flag=True, gen_len=5, batch_size=config.batch_size)
        end_time = time.perf_counter()
        logger.info(f"Time taken to evaluate GLUE benchmark: {end_time - start_time:.2f} seconds.")
        
        seval_attack_results.append(sum(toxic_val_answer_bools["seval_attack"]) / len(toxic_val_answer_bools["seval_attack"]))
        logger.info(f"seval_attack_results: {seval_attack_results[-1]}")
        seval_risk_results.append(sum(toxic_val_answer_bools["seval_risk"]) / len(toxic_val_answer_bools["seval_risk"]))
        logger.info(f"seval_risk_results: {seval_risk_results[-1]}")
        orfuzzset_results.append(1 - sum(benign_val_answer_bools["orfuzzset"]) / len(benign_val_answer_bools["orfuzzset"]))
        logger.info(f"orfuzzset_results: {orfuzzset_results[-1]}")
        nq_results.append(1 - sum(benign_val_answer_bools["nq"]) / len(benign_val_answer_bools["nq"]))
        logger.info(f"nq_results: {nq_results[-1]}")
        f1_results.append(f1_score(answer_labels, answer_results))
        logger.info("Final Evaluation Results:")
        logger.info(f"F1 Score: {f1_results[-1]}")
        glue_results = append_glue_results(glue_results, glue_result)
        logger.info(f"GLUE sst f1 score: {glue_result['sst']['f1']}")
        logger.info(f"GLUE sst MCC score: {glue_result['sst']['mcc']}")
        
        # Save temperary results to avoid losing progress
        os.makedirs(os.path.join(result_path, f"iter_{iteration}"), exist_ok=True)
        with open(os.path.join(result_path, f"iter_{iteration}", "benign_direction.pkl"), "wb") as f:
            pickle.dump(benign_direction_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "answer_direction.pkl"), "wb") as f:
            pickle.dump(answer_direction_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "benign_std.pkl"), "wb") as f:
            pickle.dump(benign_std_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "answer_std.pkl"), "wb") as f:
            pickle.dump(answer_std_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "benign_score.pkl"), "wb") as f:
            pickle.dump(benign_score_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "answer_score.pkl"), "wb") as f:
            pickle.dump(answer_score_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "benign_score_train.pkl"), "wb") as f:
            pickle.dump(benign_score_train_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "answer_score_train.pkl"), "wb") as f:
            pickle.dump(answer_score_train_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "glue_results.pkl"), "wb") as f:
            pickle.dump(glue_results, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "selected_layers_list.pkl"), "wb") as f:
            pickle.dump(selected_layers_list, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "resid_benign_dir.pkl"), "wb") as f:
            pickle.dump(benign_dir, f)
        with open(os.path.join(result_path, f"iter_{iteration}", "resid_answered_dir.pkl"), "wb") as f:
            pickle.dump(answered_dir, f)
        pd.DataFrame({
            "f1": f1_results,
            "seval_attack": seval_attack_results,
            "seval_risk": seval_risk_results,
            "nq": nq_results,
            "orfuzzset": orfuzzset_results,
        }).to_csv(os.path.join(result_path, f"iter_{iteration}", "results.csv"), index=False)
        logger.info(f"Saved results for iteration {iteration}.")
        
        # Draw and save the figure of asr/orr
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(seval_attack_results)), seval_attack_results, marker='o', label='ASR (seval_attack)')
        plt.plot(range(len(seval_risk_results)), seval_risk_results, marker='o', label='ASR (seval_risk)')
        plt.plot(range(len(orfuzzset_results)), orfuzzset_results, marker='o', label='ORR (orfuzzset)')
        plt.plot(range(len(nq_results)), nq_results, marker='o', label='ORR (nq)')
        plt.xlabel('Iteration')
        plt.ylabel('ORR/ASR')
        plt.title('Rate vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_path, "asr_orr_vs_iteration.pdf"), bbox_inches='tight')
        plt.close()
        
        # Draw and save the figure of F1 score
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(f1_results)), f1_results, marker='o', label='F1 Score')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_path, "f1_vs_iteration.pdf"), bbox_inches='tight')
        plt.close()
        
        # Draw and save the figure of GLUE results
        os.makedirs(os.path.join(figure_path, "glue_results"), exist_ok=True)
        for task, results in glue_results.items():
            if task == "perplexity":
                plt.figure(figsize=(8, 4))
                plt.plot(range(len(results["results"])), results["results"], marker='o', label='Perplexity')
                plt.xlabel('Iteration')
                plt.ylabel('Perplexity')
                plt.title(f'GLUE {task} Perplexity vs. Iteration')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(figure_path, "glue_results", f"{task}_vs_iteration.pdf"), bbox_inches='tight')
                plt.close()
                continue
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(results["f1"])), results["f1"], marker='o', label='F1 Score')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title(f'GLUE {task} F1 Score vs. Iteration')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figure_path, "glue_results", f"{task}_f1_vs_iteration.pdf"), bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(results["mcc"])), results["mcc"], marker='o', label='MCC Score')
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title(f'GLUE {task} MCC Score vs. Iteration')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figure_path, "glue_results", f"{task}_mcc_vs_iteration.pdf"), bbox_inches='tight')
            plt.close()
        
        if f1_results[-1] < 0.1 and f1_results[-1] < f1_results[0] / 2:
            logger.info("F1 score is too low. The LLM has been compromised, stopping the iteration.")
            break

    # Clear memory
    del hooked_model, hf_model, tokenizer, open_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load the model with the highest F1 score
    hf_model, hooked_model, tokenizer = load_model_and_tokenizer(model_name_dict[config.model_type], config.model_path, logger)
    open_model = OpenModel(hf_model, tokenizer, system_prompt)
    best_iter = np.argmax(f1_results) - 1 # The original model needs to be excluded
    if best_iter == -1:
        logger.info("The original model performs the best, no need to load a modified model.")
    else:
        logger.info(f"Loading the modified model with the best F1 score from iteration {best_iter}...")
        benign_direction_list = pickle.load(open(os.path.join(result_path, f"iter_{best_iter}", "benign_direction.pkl"), "rb"))
        answer_direction_list = pickle.load(open(os.path.join(result_path, f"iter_{best_iter}", "answer_direction.pkl"), "rb"))
        benign_std_list = pickle.load(open(os.path.join(result_path, f"iter_{best_iter}", "benign_std.pkl"), "rb"))
        answer_std_list = pickle.load(open(os.path.join(result_path, f"iter_{best_iter}", "answer_std.pkl"), "rb"))
        selected_layers_list = pickle.load(open(os.path.join(result_path, f"iter_{best_iter}", "selected_layers_list.pkl"), "rb"))
        hooked_model = load_modified_model(
            hooked_model,
            benign_direction_list,
            benign_std_list,
            answer_direction_list,
            answer_std_list,
            selected_layers_list,
        )
    
    # Evaluate the final model on the test set
    logger.info("Evaluating the final model on the test set...")
    answer_labels = []
    answer_results = []
    toxic_test_gen, toxic_test_answer_bools, toxic_test_answer_rates = {}, {}, {}
    for dataset_name, toxic_test in toxic_test_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=toxic_test,
            eval_type="jb",
            batch_size=config.batch_size,
        )
        toxic_test_gen[dataset_name] = g
        toxic_test_answer_bools[dataset_name] = b
        toxic_test_answer_rates[dataset_name] = r
        answer_labels.extend(len(b)*[False])
        answer_results.extend(b)
        logger.info(f"ASR for {dataset_name}: {sum(b) / len(b)}")
    benign_test_gen, benign_test_answer_bools, benign_test_answer_rates = {}, {}, {}
    for dataset_name, benign_test in benign_test_dict.items():
        g, b, r = eval_answer_rate(
            model=open_model,
            answer_eval_agent=qwen_guard_eval_agent,
            queries=benign_test,
            eval_type="or",
            batch_size=config.batch_size,
        )
        benign_test_gen[dataset_name] = g
        benign_test_answer_bools[dataset_name] = b
        benign_test_answer_rates[dataset_name] = r
        answer_labels.extend(len(b)*[True])
        answer_results.extend(b)
        logger.info(f"ORR for {dataset_name}: {1 - sum(b) / len(b)}")
    
    glue_result = {task:{metric:value[best_iter + 1] for metric, value in result_dict.items()} for task, result_dict in glue_results.items()}
    
    f1_test_result = f1_score(answer_labels, answer_results)
    logger.info("Final Evaluation Results on Test Set:")
    logger.info(f"F1 Test Result: {f1_test_result}")
    logger.info(f"GLUE sst f1 score: {glue_result['sst']['f1']}")
    logger.info(f"GLUE sst MCC score: {glue_result['sst']['mcc']}")
    
    # Save the test results
    with open(os.path.join(result_path, "test_final_results.json"), "w") as f:
        json.dump({
            "f1": f1_test_result,
            "seval_attack": sum(toxic_test_answer_bools["seval_attack"]) / len(toxic_test_answer_bools["seval_attack"]),
            "seval_risk": sum(toxic_test_answer_bools["seval_risk"]) / len(toxic_test_answer_bools["seval_risk"]),
            "orfuzzset": 1 - sum(benign_test_answer_bools["orfuzzset"]) / len(benign_test_answer_bools["orfuzzset"]),
            "nq": 1 - sum(benign_test_answer_bools["nq"]) / len(benign_test_answer_bools["nq"]),
            "glue": glue_result
        }, f, indent=4)
    
    # Save the final results
    pd.DataFrame({
        "f1": f1_results,
        "seval_attack": seval_attack_results,
        "seval_risk": seval_risk_results,
        "nq": nq_results,
        "orfuzzset": orfuzzset_results,
    }).to_csv(os.path.join(result_path, "final_results.csv"), index=False)

    # (Optional) Save the final Huggingface model (Better to use load_modified_model during experiments to save space.)
    if config.save_hf:
        logger.info("Saving the final model to Huggingface...")
        lm_model = hf_model

        state_dict = hooked_model.state_dict()
        lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())

        for l in range(hooked_model.cfg.n_layers):
            lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
                einops.rearrange(
                    state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=hooked_model.cfg.n_heads
                ).contiguous()
            )
            lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
                torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )
        os.makedirs(os.path.join(result_path, "final_model"), exist_ok=True)
        hf_model.save_pretrained(os.path.join(result_path, "final_model"))
        tokenizer.save_pretrained(os.path.join(result_path, "final_model"))