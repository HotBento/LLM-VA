import os
import re
import time
import pickle
import pandas as pd
import torch
# torch.cuda.set_per_process_memory_fraction(0.4)

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from tqdm import tqdm

from model import OpenModel, load_model_and_tokenizer, model_name_dict
from config import GeneralAbilityConfig
from utils.interpret_utils import load_modified_model

prompt_template = r"""Solve the problem and return the final answer between \answer{}. Here is an example:

Question: 3 * 4 = ?
Answer:
3 * 4 = 3 + 3 + 3 + 3 = 12
\answer{12}

Now, solve the following question:"""

question_template = """Question: {question}
Answer:"""

def extract_answer(text):
    start = text.find("\\answer{")
    end = text.find("}", start)
    if start != -1 and end != -1:
        # find the last integer between start and end
        answer = text[start + len("\\answer{"):end]
        answer = answer.replace(",", "").strip()
        answer = re.findall(r'\d+', answer)
        if answer:
            return answer[-1]  # Return the last found integer
        
    return None

if __name__ == "__main__":
    # Load config
    config = GeneralAbilityConfig.get_config()

    # Init logger
    log_path = os.path.join(config.log_path, "general_ability", config.dataset_name, config.model_type + f"-{config.suffix}" if config.modified else config.model_type)
    os.makedirs(log_path, exist_ok=True)
    logger.add(
        os.path.join(log_path, f"{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.log"),
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG" if config.debug else "INFO",
    )
    
    # Init result directory and config file
    result_path = os.path.join(config.result_path, config.dataset_name, config.model_type + f"-{config.suffix}" if config.modified else config.model_type)
    os.makedirs(result_path, exist_ok=True)
    config.save_yaml(os.path.join(result_path, "config.yaml"))

    # Load model and tokenizer
    hf_model, hooked_model, tokenizer = load_model_and_tokenizer(model_name_dict[config.model_type], config.model_path, logger)
    if config.modified:
        final_result_df = pd.read_csv(os.path.join(config.modify_path, "final_results.csv"))
        # find the iteration with maximum f1 score
        best_iter = final_result_df["f1"].idxmax() - 1
        
        if best_iter != -1:
            max_iter_path = os.path.join(config.modify_path, f"iter_{best_iter}")
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
            with open(os.path.join(max_iter_path, "answer_direction.pkl"), "rb") as f:
                answer_direction_list = pickle.load(f)
            with open(os.path.join(max_iter_path, "answer_std.pkl"), "rb") as f:
                answer_std_list = pickle.load(f)
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
    m = OpenModel(hf_model, tokenizer, None)
    
    # Load dataset
    if config.dataset_name == "aime":
        dataset = load_dataset("yentinglin/aime_2025", "default")
        question = dataset["train"]["problem"]
        target = dataset["train"]["answer"]
    elif config.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
        question = dataset["train"]["question"][:500]
        target = list(map(lambda x: x.split("####")[-1].strip().replace(",", ""), dataset["train"]["answer"]))[:500]
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
    # Evaluation
    gen = []
    for i in tqdm(range(0, len(question), config.batch_size)):
        batch = question[i:i + config.batch_size]
        batched_inputs = [prompt_template + question_template.format(question=q) for q in batch]
        gen.extend(m.generate(batched_inputs, max_new_tokens=2048, do_sample=config.do_sample))
    
    answers = [extract_answer(g) for g in gen]
    judge_results = [(a != None and a == b) for a, b in zip(answers, target)]
    logger.info(f"Total: {len(judge_results)}, Correct: {sum(judge_results)}, Accuracy: {sum(judge_results) / len(judge_results) if judge_results else 0:.2f}")

    # Save results
    result_df = pd.DataFrame({
        "input": question,
        "target": target,
        "generated_text": gen,
        "answer": answers,
        "correct": judge_results
    })
    result_df.to_csv(os.path.join(result_path, "results.csv"), index=False)
