import plotly.express as px
import plotly.io as pio
import pandas as pd
from .prompt import SYS_PROMPT
from typing import List, Optional, Union
from circuitsvis.attention import attention_heads
from transformer_lens import ActivationCache, HookedTransformer
from jaxtyping import Float
from tqdm import tqdm
import transformer_lens.utils as utils
import torch
import einops
from torch import Tensor
import matplotlib.pyplot as plt

from transformers import (
    PreTrainedTokenizer,
)

from matplotlib.colors import LinearSegmentedColormap, Normalize

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()


def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()

def visualize_attention_patterns(
    model: HookedTransformer,
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
    token_slice:slice|None=None,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=0
    )
    if token_slice != None:
        patterns = patterns[:, token_slice, token_slice]
        str_tokens = str_tokens[token_slice]

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"

def get_single_chat(input_str:str, model_type:str, tokenizer:PreTrainedTokenizer, interpret:bool=False):
    if interpret:
        if SYS_PROMPT[model_type] != None:
            chat_list = [
                {"role" : "system", "content" : SYS_PROMPT[model_type]},
                {"role" : "user", "content" :  SYS_PROMPT["interpret"].format(input=input_str)},
            ]
        else:
            chat_list = [
                {"role" : "user", "content" :  SYS_PROMPT["interpret"].format(input=input_str)},
            ]
    else:
        if SYS_PROMPT[model_type] != None:
            chat_list = [
                {"role" : "system", "content" : SYS_PROMPT[model_type]},
                {"role" : "user", "content" : input_str},
            ]
        else:
            chat_list = [
                {"role" : "user", "content" :  input_str},
            ]
    chat = tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)
    return chat

def split_df(df:pd.DataFrame, column:str, sign:str):
    df_include = df[df[column].str.contains(sign)].reset_index(drop=True)
    df_not_include = df[~df[column].str.contains(sign)].reset_index(drop=True)
    return df_include, df_not_include

def get_modified_matrix(
    matrix: Float[Tensor, "... d_model"],
    benign_direction: Float[Tensor, "d_act"],
    benign_std: float,
    answer_direction: Float[Tensor, "d_act"],
    answer_std: float,
) -> Float[Tensor, "... d_model"]:
    """
    Modify the given matrix with respect to the benign and answer directions, scaling by their standard deviations.
    Args:
        matrix: The matrix to be orthogonalized.
        benign_direction: The benign direction vector.
        benign_std: The standard deviation for the benign direction.
        answer_direction: The answer direction vector.
        answer_std: The standard deviation for the answer direction.
    Returns:
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
    Args:
        hooked_model: The original HookedTransformer model to be modified.
        benign_direction_list: List of benign direction vectors for each iteration and each layer.
        benign_std_list: List of standard deviations for the benign directions for each iteration and each layer.
        answer_direction_list: List of answer direction vectors for each iteration and each layer.
        answer_std_list: List of standard deviations for the answer directions for each iteration and each layer.
    Returns:
        The HookedTransformer model with the modified weights (IN PLACE).
    """
    for benign_activation, benign_std_dict, answer_activation, answer_std_dict, selected_layers in zip(benign_direction_list, benign_std_list, answer_direction_list, answer_std_list, selected_layer_list):
        for i, block in tqdm(enumerate(hooked_model.blocks)):
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