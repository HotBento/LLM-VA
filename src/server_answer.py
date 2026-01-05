import torch
from flask import Flask, request, jsonify
from eval import QwenAnswerEvaluator, QwenToxicEvaluator, QwenGuardEvaluator
import threading
import queue
import time
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Llama4ForConditionalGeneration, AutoProcessor
from transformers import QuantoConfig, AutoConfig
import argparse
from config import ServerConfig
from loguru import logger

app = Flask(__name__)

# Background thread to process the request queue
def process_requests():
    while True:
        try:
            # Get request from the queue
            request_id, evaluator_type, data = request_queue.get()
            if evaluator_type == "guard":
                inputs = data["inputs"]
                outputs = data["outputs"]
                eval_type = data["eval_type"]
                results, predictions = qwen_guard_evaluator.evaluate(inputs, outputs, eval_type)
            else:
                results, predictions = None, None

            # Store the result in the response dictionary
            response_dict[request_id] = {"results": results, "predictions": predictions}
        except Exception as e:
            response_dict[request_id] = {"error": str(e)}
            logger.error(f"Error processing request {request_id}: {e}")
        finally:
            request_queue.task_done()

@app.route('/evaluate/guard', methods=['POST'])
def evaluate_guard():
    try:
        data = request.json
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        eval_type = data.get("eval_type", [])

        if not inputs or not outputs:
            return jsonify({"error": "inputs and outputs are required"}), 400

        # Generate a unique request ID
        request_id = f"{next(request_counter)}_guard"
        # Put the request into the queue
        request_queue.put((request_id, "guard", {"inputs": inputs, "outputs": outputs, "eval_type": eval_type}))
        logger.info(f"Received guard evaluation request: {request_id}, inputs: {inputs[0]}, outputs: {outputs[0]}")

        # Wait for the result
        while request_id not in response_dict:
            time.sleep(0.1)

        # Return the result
        result = response_dict.pop(request_id)
        logger.info(f"Returning guard evaluation result for request: {result['results'][0]}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in guard evaluation request: {e}")
        return jsonify({"error": str(e)}), 500



# Clear the cached GPU memory
@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        torch.cuda.empty_cache()
        return jsonify({"message": "GPU cache cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    config = ServerConfig.get_config()

    request_counter = itertools.count()

    quantization_config = QuantoConfig(weights="int8")

    qwen_guard_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3Guard-Gen-8B", device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
    qwen_guard_processor = AutoTokenizer.from_pretrained("Qwen/Qwen3Guard-Gen-8B", padding_side="left")

    # Initialize the evaluator
    qwen_guard_evaluator = QwenGuardEvaluator(logger=logger)
    qwen_guard_evaluator.model = qwen_guard_model
    qwen_guard_evaluator.tokenizer = qwen_guard_processor

    # Request queue
    request_queue = queue.Queue()
    response_dict = {}
    # Start the background thread
    threading.Thread(target=process_requests, daemon=True).start()
    app.run(host=config.host, port=config.port)