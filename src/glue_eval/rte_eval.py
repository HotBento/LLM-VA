from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np
import os
import gc

MAX_NUMBER_OF_FEW_SHOTS = 100
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)

class RTEEval():

    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split(os.path.join(CURRENT_DIR, 'dataset/rte.pkl'), number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = ''
        self.postfix_prompt = 'answer:'
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{few_shot['sentence1']}\nquestion: {few_shot['sentence2']} True or False?\nanswer: {'False' if few_shot['label'] == 0 else 'True'}\n")

    # def _create_prompt(self, example):
    #     prompt = example['sentence1'] + '\n'
    #     prompt += 'question: ' + example['sentence2'] + ' True or False?'  + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt

    def _create_prompt(self, example, gen_len):
        prompt = example['sentence1'] + '\n'
        prompt += 'question: ' + example['sentence2'] + ' True or False?'  + '\n'
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
        return input_prompt
    
    def _get_answer(self, generated_text):
        answer_text = generated_text.split('answer:')[-1].strip().strip()

        if 'false' in answer_text.lower():
            return 0
        elif 'true' in answer_text.lower():
            return 1

        return -1


    def evaluate(self, gen_len = 3, print_logs = False, batch_size=16, **kwargs):
        true_tok, false_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['True', 'False'])
        
        if 'llama' in self.model.config._name_or_path.lower():
            true_tok = true_tok[1:]
            false_tok = false_tok[1:]

        true_len, false_len = (len(n) for n in [true_tok, false_tok])
        suffixes = {0: ['True', true_tok, true_len], 1: ['False', false_tok, false_len]}

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
        # batch_size = 8  # Adjust batch size based on your GPU memory
        
        for i in range(0, len(self.eval_dataset), batch_size):
            batch = self.eval_dataset[i:min(i+batch_size, len(self.eval_dataset))]
            
            # Prepare batch data
            batch_prompts = []
            batch_elements = []
            batch_prefix_tok_lens = []
            
            for element in batch:
                input_prompt = self._create_prompt(element, gen_len)
                batch_prompts.append(input_prompt)
                batch_elements.append(element)
                
                prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
                if 'llama' in self.model.config._name_or_path.lower():
                    prefix_tok_len = prefix_tok_len - 1
                batch_prefix_tok_lens.append(prefix_tok_len)
            
            # Tokenize batch
            batch_input_ids = self.tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
            
            # Generate for batch
            max_len = batch_input_ids.input_ids.shape[1] + gen_len
            batch_outputs = self.model.generate(
                batch_input_ids.input_ids,
                attention_mask=batch_input_ids.attention_mask,
                max_length=max_len,
                do_sample=False, 
                **kwargs
            )
            
            # Process each result in the batch
            for j, (element, output, input_prompt, prefix_tok_len) in enumerate(
            zip(batch_elements, batch_outputs, batch_prompts, batch_prefix_tok_lens)
            ):
                label = element['label']
                sentence1 = element['sentence1']
                sentence2 = element['sentence2']
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                input_prompt_text = self.tokenizer.decode(
                    self.tokenizer.encode(input_prompt, return_tensors='pt')[0], 
                    skip_special_tokens=True
                )
                
                # Get the model's answer
                answer = self._get_answer(generated_text)
                predictions.append(answer)
                labels.append(label)
            
                # Calculate probabilities for True/False tokens
                probs = [0 for _ in suffixes.keys()]
                gen_texts = [0 for _ in suffixes.keys()]
                
                # Create batched prompts for suffix probabilities
                suffix_prompts = [f"{input_prompt} {suffixes[i][0]}" for i in suffixes.keys()]
                suffix_tokens = self.tokenizer(suffix_prompts, return_tensors="pt", padding=True).to('cuda')
                
                with torch.no_grad():
                    suffix_logits = self.model(**suffix_tokens).logits
                if torch.cuda.memory_reserved() / (1024 * 1024) > 35000:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                for i in range(len(suffixes.keys())):
                    cur_len = suffixes[i][2]
                    
                    if "llama" in self.model.config._name_or_path.lower():
                        cur_logits = suffix_logits[i, 1:, :]
                    else:
                        cur_logits = suffix_logits[i]
                    
                    for j_pos in range(cur_len):
                        cur_tok = suffixes[i][1][j_pos]
                        probs[i] += -torch.nn.functional.log_softmax(
                            cur_logits[-cur_len + j_pos - 1, :], dim=0
                        )[cur_tok].item()
                    probs[i] /= cur_len
                    
                    gen_texts[i] = self.tokenizer.decode(cur_logits[prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim=-1))
                
                prob_yes = np.exp(-probs[0])
                prob_no = np.exp(-probs[1])
                
                answer_new = 1 if prob_yes > prob_no else 0
                predictions_new.append(answer_new)
                
                # Update counters
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
                
                # Store results
                exp_temp_dict = {
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    'label': 'True' if label == 1 else 'False',
                    'input_prompt': input_prompt_text,
                    'generated_text': generated_text.replace(input_prompt_text, ''),
                    'answer': answer,
                    'correct': answer == label,
                    'prob_yes': prob_yes,
                    'prob_no': prob_no,
                    'highest_probability_answer': 'True' if answer_new == 1 else 'False',
                    'correct_new': answer_new == label,                
                }
                stored_generations.append(exp_temp_dict)
                
                # Print progress if requested
                if print_logs:
                    mcc = matthews_corrcoef(labels, predictions)
                    f1 = f1_score(labels, predictions, average='weighted')
                    print(f"Processed {len(predictions)}/{len(self.eval_dataset)} | Acc: {correct/(correct+incorrect+invalid):.4f} | MCC: {mcc:.4f}")
        
        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': len(predictions),
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }

        return result_dict, stored_generations

if __name__ == '__main__':
    '''dataset = load_dataset("glue", "rte")
    eval_dataset = dataset['train']

    count = 0
    for example in eval_dataset:
        # print(example)
        # print()

    exit()'''


    # Load the tokenizer and model
    model_name = 'EleutherAI/gpt-j-6b'
    #model_name = 'gpt2-xl'
    #model_name = '/data/akshat/lingua-models/Llama-2-7b-hf'
    # print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    rte_eval = RTEEval(model, tokenizer)
    rte_eval.evaluate(print_logs='True')