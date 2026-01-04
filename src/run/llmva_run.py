# %%
import atexit
import signal
import subprocess
import sys
import os
from threading import Thread
import queue
from loguru import logger

import psutil

pid = os.getpid()

layer_num_dict = {
    "llama3": 32,
    "gemma": 42,
    "mistral": 32,
    "phi": 32,
    "qwen2-3b":36,
    "qwen": 28,
    "qwen2-14b": 48,
    "qwen3-4b": 36,
    "qwen3-8b": 36,
    "qwen3-14b":40,
    "phi4-mini": 32,
    "phi4": 32,
}


model_name_dict = {
    "llama3":"meta-llama/Llama-3.1-8B-Instruct",
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
# %%
model_list = ["llama3", "gemma", "mistral", "phi", "qwen3-8b", "phi4-mini", "qwen2-3b", "qwen3-4b", "qwen", "qwen2-14b", "qwen3-14b", "phi4"]
max_iter = 30
select_layer_num_list = [30, 36, 42, 48, 54, 60]

# %%
command_list = []

for m in model_list:
    next_model = False
    for select_layer_num in select_layer_num_list:
        if select_layer_num > 2 * layer_num_dict[m]:
            select_layer_num = 2 * layer_num_dict[m]
            next_model = True
        if os.path.exists(f"result/llmva-{select_layer_num}/{m}/final_results.csv"):
            logger.info(f"Skip {m} select {select_layer_num} because result already exists")
            continue
        batch_size = 8
        add_generation_prompt = True
        command = f"python src/llmva.py --result_path result/llmva-{select_layer_num} --figure_path figure/llmva-{select_layer_num} --max_iter {max_iter} --model_type {m} --model_path {model_name_dict[m]} --log_path log/llmva/{m} --debug --select_layer_num {select_layer_num} --batch_size {batch_size} --select_token_num 1 --n_samples 256 --add_generation_prompt {add_generation_prompt} --ppid {pid}"
        command_list.append(command)
        if next_model:
            break


# %%
logger.info(f"Total commands to run: {command_list}")

# %%
cuda_id_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
cuda_id_list = cuda_id_list.split(",")
cuda_id_list = [int(id.strip()) for id in cuda_id_list]
process_per_device = 1

cuda_id_dict = {
    cuda_id: process_per_device for cuda_id in cuda_id_list
}
if_end = False

class Worker():
    def __init__(self, cuda_id):
        self.cuda_id = cuda_id
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        while not command_queue.empty():
            global if_end
            if if_end:
                break
            command = command_queue.get()
            full_command = f"CUDA_VISIBLE_DEVICES=\"{self.cuda_id}\" " + command
            logger.info(f"Running command on GPU {self.cuda_id}: {command}")
            try:
                process = subprocess.run(full_command, shell=True)
                running_processes.append(process)
                return_code = process.returncode
                if return_code != 0:
                    logger.error(f"Command failed with return code {return_code} on GPU {self.cuda_id}: {command}")
                    command_list.append(command)
            except Exception as e:
                logger.error(f"Exception occurred while running command on GPU {self.cuda_id}: {command}. Exception: {e}")
                command_list.append(command)
            finally:
                if process in running_processes:
                    running_processes.remove(process)
    def end(self):
        self.thread.join(0.1)

def cleanup():
    global running_processes
    global if_end
    global workers
    for worker in workers:
        worker.end()
    if_end = True
    logger.info("Cleaning up running processes...")
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and f"--ppid {pid}" in ' '.join(cmdline):
                logger.info(f"Killing process {proc.pid} running: {' '.join(cmdline)}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for process in running_processes:
        try:
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating process with PID {process.pid}")
                process.terminate()
                process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating process with PID {process.pid}: {e}")

def sig_handler(signum, frame):
    cleanup()
    sys.exit(0)
atexit.register(cleanup)
signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)
while all([ v > 0 for v in cuda_id_dict.values()]) and not if_end:
    command_queue = queue.Queue()
    for cmd in command_list:
        command_queue.put(cmd)
    command_list = []
    running_processes = []
    workers = []
    for cuda_id, capacity in cuda_id_dict.items():
        for _ in range(capacity):
            worker = Worker(cuda_id)
            workers.append(worker)
    for worker in workers:
        worker.thread.join()
    if command_list:
        logger.info(f"Retrying {len(command_list)} failed commands...")
        cuda_id_dict = {
            k:v-1 for k,v in cuda_id_dict.items() if v > 0
        }
    else:
        break

