from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from glue_eval.useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import time
import torch
import numpy as np
import os
import gc

MAX_NUMBER_OF_FEW_SHOTS = 100
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)

class SSTEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split(os.path.join(CURRENT_DIR, 'dataset/sst2.pkl'), number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Review :'
        self.postfix_prompt = '\nSentiment :'
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt} {few_shot['sentence']}{self.postfix_prompt} {'positive' if few_shot['label']==1 else 'negative'}\n")

    def _create_prompt(self, example, gen_len):
        question = self.prefix_prompt + example['sentence'] + self.postfix_prompt
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
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Sentiment :')[-1].strip().strip()

        if 'positive' in answer_text.lower():
            return 1
        elif 'negative' in answer_text.lower():
            return 0

        return -1


    def evaluate(self, gen_len = 3, print_logs = False, batch_size=16, **kwargs):
        pos_tok, neg_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['positive', 'negative'])

        if 'llama' in self.model.config._name_or_path.lower():
            pos_tok = pos_tok[1:]
            neg_tok = neg_tok[1:]

        pos_len, neg_len = (len(n) for n in [pos_tok, neg_tok])
        suffixes = {0: ['positive', pos_tok, pos_len], 1: ['negative', neg_tok, neg_len]}

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
        # batch_size = 16  # You can adjust this based on your GPU memory
        
        # Process all examples in batches
        for i in range(0, len(self.eval_dataset), batch_size):
            batch_examples = self.eval_dataset[i:i+batch_size]
            actual_batch_size = len(batch_examples)
            
            # Prepare batch inputs for generation
            batch_inputs = []
            batch_sentences = []
            batch_labels = []
            
            for example in batch_examples:
                input_prompt, sentence, label = self._create_prompt(example, gen_len)
                batch_inputs.append(input_prompt)
                batch_sentences.append(sentence)
                batch_labels.append(label)
            
            # Tokenize all inputs in batch
            batch_input_ids = self.tokenizer(batch_inputs, return_tensors='pt', padding=True).to('cuda')
            
            # Generate outputs for the batch
            max_len = batch_input_ids.input_ids.shape[1] + gen_len
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    batch_input_ids.input_ids,
                    attention_mask=batch_input_ids.attention_mask,
                    max_length=max_len,
                    do_sample=False, 
                    **kwargs
                )
            
            # Decode the generated texts
            batch_generated_texts = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            
            # Process each example's results
            for j in range(actual_batch_size):
                s = i + j  # Update current example index
                
                generated_text = batch_generated_texts[j]
                input_prompt = batch_inputs[j]
                sentence = batch_sentences[j]
                label = batch_labels[j]
                
                # Get answer from generated text
                answer = self._get_answer(generated_text)
                predictions.append(answer)
                labels.append(label)
                
                # Calculate prefix token length
                prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])
                if 'llama' in self.model.config._name_or_path.lower():
                    prefix_tok_len = prefix_tok_len - 1
                
                # Calculate probabilities for each suffix
                probs = [0 for _ in suffixes.keys()]
                gen_texts = [0 for _ in suffixes.keys()]
                
                # Create batched prompts for suffix probabilities
                suffix_prompts = [f"{input_prompt} {suffixes[i][0]}" for i in suffixes.keys()]
                suffix_tokens = self.tokenizer(suffix_prompts, return_tensors="pt", padding=True).to('cuda')
                
                with torch.no_grad():
                    suffix_logits = self.model(**suffix_tokens, **kwargs).logits
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


                prob_pos = np.exp(-probs[0])
                prob_neg = np.exp(-probs[1])

                # print(f"prob_positive: {prob_pos}, prob_negative: {prob_neg}")

                answer_new = 1 if prob_pos > prob_neg else 0
                predictions_new.append(answer_new)
                # print(f"prediction: {answer}, true: {label}")

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
                    'input_prompt': input_prompt,
                    'true_answer': 'positive' if label == 1 else 'negative',  
                    'generated_text': generated_text.replace(input_prompt, ''),
                    'answer': answer,
                    'correct': answer == label,
                    'prob_positive': prob_pos,
                    'prob_negative': prob_neg,
                    'highest_probability_answer': 'positive' if answer_new == 1 else 'negative',
                    'correct_new': answer_new == label,
                }
                stored_generations.append(exp_temp_dict)

                if print_logs:
                    mcc = matthews_corrcoef(labels, predictions)
                    f1 = f1_score(labels, predictions, average='weighted')
                    # print(generated_text)
                    # print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                    # print('--'*50)


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
    sst_eval = SSTEval(None, None)
    exit()
    model_name = 'EleutherAI/gpt-j-6b'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')

    sst_eval = SSTEval(model, tokenizer)
    correct, incorrect, invalid, total = sst_eval.evaluate(print_logs='True')