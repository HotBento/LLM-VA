import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from glue_eval.sst_eval import SSTEval
from glue_eval.mrpc_eval import MRPCEval
from glue_eval.cola_eval import COLAEval
from glue_eval.rte_eval import RTEEval
from glue_eval.mmlu_eval import MMLUEval
from glue_eval.sentiment_analysis_eval import SENTIMENT_ANALYSIS_Eval
from glue_eval.dialogue_eval import DIALOGUE_Eval
from glue_eval.nli_eval import NLIEval
# from util.perplexity import perplexity
from datasets import load_dataset
from loguru import logger
import time
import gc

def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()

class GLUEEval():
    def __init__(self, model, tokenizer, number_of_tests = None, sst_number_of_few_shots = 2, mrpc_number_of_few_shots = 2, cola_number_of_few_shots = 2, rte_number_of_few_shots = 2, mmlu_number_of_few_shots = 2, sentiment_analysis_number_of_few_shots = 2, nli_number_of_few_shots = 2, dialogue_number_of_few_shots = 2):
        self.model = model

        self.tokenizer = tokenizer

        self.sst_eval = SSTEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sst_number_of_few_shots)

        self.mrpc_eval = MRPCEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mrpc_number_of_few_shots)

        self.cola_eval = COLAEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = cola_number_of_few_shots)

        self.rte_eval = RTEEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = rte_number_of_few_shots)

        self.mmlu_eval = MMLUEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mmlu_number_of_few_shots)

        self.sentiment_analysis_eval = SENTIMENT_ANALYSIS_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sentiment_analysis_number_of_few_shots)

        self.nli_eval = NLIEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = nli_number_of_few_shots)

        self.dialogue_eval = DIALOGUE_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = dialogue_number_of_few_shots)


    # def _save_generations(self, record_path, generations, task):
    #     #store individual generation file
    #     output_filename = record_path.replace('.json', '_' + task + '_gen.json')
    #     with open(output_filename, "w") as f:
    #         json.dump(generations, f, indent=4)


    @torch.no_grad()
    def evaluate(self, perplexity_flag = False, sst_flag = False, mmlu_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, nli_flag = False, sentiment_analysis_flag = False, dialogue_flag = False, gen_len = 5, batch_size=16, **kwargs):
        glue_results = {}
        if perplexity_flag:
            while True:
                try:
                    raw_ds = load_dataset(
                                "wikitext",
                                dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")["wikitext"],
                                )
                    break
                except Exception as e:
                    logger.warning(f"Error loading dataset: {e}")
                    time.sleep(1)
            glue_results['perplexity'] = perplexity(self.model, self.tokenizer, " ".join(raw_ds["train"]['text'][:20]), max_input_length=100)
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['sst'] = result_dict
            # logger.info(f"SST Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'sst')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if mmlu_flag:
            result_dict, generations = self.mmlu_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['mmlu'] = result_dict
            # logger.info(f"MMLU Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'mmlu')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['mrpc'] = result_dict
            # logger.info(f"MRPC Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'mrpc')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['cola'] = result_dict
            # logger.info(f"COLA Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'cola')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['rte'] = result_dict
            # logger.info(f"RTE Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'rte')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if sentiment_analysis_flag:
            result_dict, generations = self.sentiment_analysis_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['sentiment_analysis'] = result_dict
            # logger.info(f"Sentiment Analysis Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'sentiment_analysis')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if nli_flag:
            result_dict, generations = self.nli_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['nli'] = result_dict
            # logger.info(f"NLI Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'nli')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        if dialogue_flag:
            result_dict, generations = self.dialogue_eval.evaluate(gen_len, batch_size=batch_size, **kwargs)
            glue_results['dialogue'] = result_dict
            # logger.info(f"Dialogue Generation Sample: {generations[0]}")
            # self._save_generations(record_path, generations, 'dialogue')
        # gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        # logger.info(f"当前GPU占用: {gpu_memory:.2f} MB, 保留: {gpu_reserved:.2f} MB")
        gc.collect()
        torch.cuda.empty_cache()
        return glue_results


        

