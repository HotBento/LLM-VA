from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import math
import torch
import time
import numpy as np
import os
import gc

MAX_NUMBER_OF_FEW_SHOTS = 100
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)

class COLAEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split(os.path.join(CURRENT_DIR, 'dataset/cola.pkl'), number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Is this sentence linguistically acceptable?\n'

        self.postfix_prompt = 'Answer: '
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt}Sentence: {few_shot['sentence']}\nAnswer: {'No' if few_shot['label'] == 0 else 'Yes'}\n")

    
    # def _create_prompt(self, example):
    #     prompt = 'Sentence: ' + example['sentence'] + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt, example['sentence'], example['label']

    def _create_prompt(self, example, gen_len):
        prompt = 'Sentence: ' + example['sentence'] + '\n'
        question = self.prefix_prompt + prompt + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = 2048 - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        # print(type(example['label']))
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Answer:')[-1].strip().strip()

        if 'yes' in answer_text.lower():
            return 1
        elif 'no' in answer_text.lower():
            return 0

        return -1

    def evaluate(self, gen_len = 10, print_logs = False, batch_size=16, **kwargs):

        yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Yes', 'No'])

        if "llama" in self.model.config._name_or_path.lower():
            yes_tok = yes_tok[1:]
            no_tok = no_tok[1:]

        yes_len, no_len = (len(n) for n in [yes_tok, no_tok])

        suffixes = {0: ['Yes', yes_tok, yes_len], 1: ['No', no_tok, no_len]}

        correct = 0
        incorrect = 0
        invalid = 0


        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()

        for s in range(0, len(self.eval_dataset), batch_size):
            batch_examples = self.eval_dataset[s:s+batch_size]
            real_size = len(batch_examples)
            
            # Prepare batch inputs
            batch_inputs = []
            batch_sentences = []
            batch_labels = []
            batch_prefix_tok_lens = []
            
            for example in batch_examples:
                input_prompt, sentence, label = self._create_prompt(example, gen_len)
                batch_inputs.append(input_prompt)
                batch_sentences.append(sentence)
                batch_labels.append(label)
                
                prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
                if 'llama' in self.model.config._name_or_path.lower():
                    prefix_tok_len = prefix_tok_len - 1
                batch_prefix_tok_lens.append(prefix_tok_len)
            
            # Tokenize all inputs at once
            tokenized_inputs = self.tokenizer(batch_inputs, padding=True, return_tensors='pt').to('cuda')
            
            # Generate text for the entire batch at once
            max_len = tokenized_inputs.input_ids.shape[1] + gen_len
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    tokenized_inputs.input_ids, 
                    attention_mask=tokenized_inputs.attention_mask,
                    max_length=max_len, 
                    do_sample=False,
                    return_dict_in_generate=False,
                    **kwargs
                )
            
            # Process results for the batch
            for i in range(real_size):
                if s + i >= len(self.eval_dataset):
                    break
                    
                input_prompt = batch_inputs[i]
                sentence = batch_sentences[i]
                label = batch_labels[i]
                prefix_tok_len = batch_prefix_tok_lens[i]
                
                input_prompt_text = self.tokenizer.decode(tokenized_inputs.input_ids[i], skip_special_tokens=True)
                generated_text = self.tokenizer.decode(batch_outputs[i], skip_special_tokens=True)
                answer = self._get_answer(generated_text)
                
                predictions.append(answer)
                labels.append(label)
                
                # Calculate probabilities for Yes/No answers
                batch_prompt_yes_no = [
                    f"{input_prompt} {suffixes[0][0]}",
                    f"{input_prompt} {suffixes[1][0]}"
                ]
                prompt_tok = self.tokenizer(batch_prompt_yes_no, padding=True, return_tensors="pt").to('cuda')
                
                with torch.no_grad():
                    logits = self.model(**prompt_tok, **kwargs).logits
                if torch.cuda.memory_reserved() / (1024 * 1024) > 35000:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]
                
                probs = [0, 0]
                for j in range(2):  # Yes and No
                    cur_len = suffixes[j][2]
                    for k in range(cur_len):
                        cur_tok = suffixes[j][1][k]
                        probs[j] += -torch.nn.functional.log_softmax(
                            # logits[j, prefix_tok_len + k - 1, :], dim=0
                            logits[j, -cur_len + k - 1, :], dim=0
                        )[cur_tok].item()
                    probs[j] /= cur_len
                
                prob_yes = np.exp(-probs[0])
                prob_no = np.exp(-probs[1])
                
                answer_new = 1 if prob_yes > prob_no else 0
                predictions_new.append(answer_new)
                
                if answer == -1:
                    invalid += 1
                else:
                    if answer == label:
                        correct += 1
                        if label == 1:
                            pos_correct += 1
                        elif label == 0:
                            neg_correct += 1
                    else:
                        incorrect += 1
                        if label == 1:
                            pos_incorrect += 1
                        elif label == 0:
                            neg_incorrect += 1
                
                exp_temp_dict = {
                    'sentence': sentence,
                    'input_prompt': input_prompt_text,
                    'true_answer': 'Yes' if label == 1 else 'No',
                    'generated_text': generated_text.replace(input_prompt_text, ''),
                    'answer': answer,
                    'correct': answer == label,
                    'prob_yes': prob_yes,
                    'prob_no': prob_no,
                    'highest_probability_answer': 'Yes' if answer_new == 1 else 'No',
                    'correct_new': answer_new == label,
                }
                stored_generations.append(exp_temp_dict)
                
                if print_logs and (s + i + 1) % 10 == 0:
                    mcc = matthews_corrcoef(labels, predictions)
                    f1 = f1_score(labels, predictions, average='weighted')

        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }
        return result_dict, stored_generations

if __name__ == '__main__':
    # Load the tokenizer and model
    model_name = "/data/akshat/models/gpt2-xl"
    #model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    cola_eval = COLAEval(model, tokenizer)
    result_dict, stored_generations = cola_eval.evaluate(print_logs=True)