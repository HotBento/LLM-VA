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

# 后台线程处理队列
def process_requests():
    while True:
        try:
            # 从队列中获取请求
            request_id, evaluator_type, data = request_queue.get()
            # if evaluator_type == "answer":
            #     inputs = data["inputs"]
            #     outputs = data["outputs"]
            #     th = data["th"]
            #     results, predictions = qwen_answer_evaluator.evaluate(inputs, outputs, th=th)
            if evaluator_type == "guard":
                inputs = data["inputs"]
                outputs = data["outputs"]
                eval_type = data["eval_type"]
                results, predictions = qwen_guard_evaluator.evaluate(inputs, outputs, eval_type)
            else:
                results, predictions = None, None

            # 将结果存入响应字典
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

        # 生成唯一请求 ID
        request_id = f"{next(request_counter)}_guard"
        # 将请求放入队列
        request_queue.put((request_id, "guard", {"inputs": inputs, "outputs": outputs, "eval_type": eval_type}))
        logger.info(f"Received guard evaluation request: {request_id}, inputs: {inputs[0]}, outputs: {outputs[0]}")

        # 等待结果
        while request_id not in response_dict:
            time.sleep(0.1)

        # 返回结果
        result = response_dict.pop(request_id)
        logger.info(f"Returning guard evaluation result for request: {result['results'][0]}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in guard evaluation request: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route('/evaluate/answer', methods=['POST'])
# def evaluate_answer():
#     try:
#         data = request.json
#         inputs = data.get("inputs", [])
#         outputs = data.get("outputs", [])
#         th = data.get("th", 0.5)

#         if not inputs or not outputs:
#             return jsonify({"error": "inputs and outputs are required"}), 400

#         # 生成唯一请求 ID
#         request_id = f"{next(request_counter)}_answer"
#         # 将请求放入队列
#         request_queue.put((request_id, "answer", {"inputs": inputs, "outputs": outputs, "th": th}))

#         # 等待结果
#         while request_id not in response_dict:
#             time.sleep(0.1)

#         # 返回结果
#         return jsonify(response_dict.pop(request_id))
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Clear the cached GPU memory
@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        torch.cuda.empty_cache()
        return jsonify({"message": "GPU cache cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/evaluate/toxic', methods=['POST'])
# def evaluate_toxic():
#     try:
#         data = request.json
#         inputs = data.get("inputs", [])
#         th = data.get("th", 0.5)

#         if not inputs:
#             return jsonify({"error": "inputs are required"}), 400

#         # 生成唯一请求 ID
#         request_id = f"{next(request_counter)}_toxic"
#         # 将请求放入队列
#         request_queue.put((request_id, "toxic", {"inputs": inputs, "th": th}))

#         # 等待结果
#         while request_id not in response_dict:
#             time.sleep(0.1)

#         # 返回结果
#         return jsonify(response_dict.pop(request_id))
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    config = ServerConfig.get_config()

    request_counter = itertools.count()

    quantization_config = QuantoConfig(weights="int8")
    # answer_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained("model_finetuning/lora/qwen_answer", torch_dtype="auto",quantization_config=quantization_config, device_map="cuda:0", attn_implementation="eager").eval()
    # toxic_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained("model_finetuning/lora/qwen_answer", torch_dtype="auto",quantization_config=quantization_config, device_map="cuda:0", attn_implementation="eager").eval()
    # eval_tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side="left")
    
    # llama_guard_config = AutoConfig.from_pretrained("Qwen/Qwen3Guard-Gen-8B")
    # llama_guard_config.text_config.attention_chunk_size = 8192
    qwen_guard_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3Guard-Gen-8B", device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
    qwen_guard_processor = AutoTokenizer.from_pretrained("Qwen/Qwen3Guard-Gen-8B", padding_side="left")

    # 初始化评估器
    qwen_guard_evaluator = QwenGuardEvaluator(logger=logger)
    qwen_guard_evaluator.model = qwen_guard_model
    qwen_guard_evaluator.tokenizer = qwen_guard_processor

    # 请求队列
    request_queue = queue.Queue()
    response_dict = {}
    # 启动后台线程
    threading.Thread(target=process_requests, daemon=True).start()
    app.run(host=config.host, port=config.port)