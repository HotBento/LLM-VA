# %%
import atexit
import signal
import subprocess
import sys
sys.path.append("./src")
import os
from threading import Semaphore, Thread
from loguru import logger

import psutil

pid = os.getpid()

from config.layer_settings import select_layer_num_dict_st as layer_num_dict


# %%
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
model_list = ["llama3", "gemma", "mistral", "phi", "qwen2-3b", "qwen", "qwen2-14b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "phi4-mini", "phi4"]

dataset_list = ["aime", "gsm8k"]
batch_size = 8


# %%
command_list = []
for dataset in dataset_list:
    for m in model_list:
        if os.path.exists(f"result_1215/general_ability/{dataset}/{m}/results.csv"):
            logger.info(f"Skip {m} in {dataset} because result already exists")
            continue
        command = f"python src/general_eval.py --dataset_name {dataset} --model_type {m} --model_path {model_name_dict[m]} --result_path result_1215/general_ability --batch_size {batch_size} --do_sample --ppid {pid}"
        command_list.append(command)

for dataset in dataset_list:
    for m in model_list:
        if os.path.exists(f"result_1215/general_ability/{dataset}/{m}-modified/results.csv"):
            logger.info(f"Skip {m}-modified in {dataset} because result already exists")
            continue
        command = f"python src/general_eval.py --dataset_name {dataset} --model_type {m} --model_path {model_name_dict[m]} --result_path result_1215/general_ability --batch_size {batch_size} --do_sample --modified --modify_path result_1215/lvs_select_st{layer_num_dict[m]}/{m} --ppid {pid}"
        command_list.append(command)

for dataset in dataset_list:
    for m in model_list:
        if os.path.exists(f"result_1215/general_ability/{dataset}/{m}-random/results.csv"):
            logger.info(f"Skip {m}-random in {dataset} because result already exists")
            continue
        command = f"python src/general_eval.py --dataset_name {dataset} --model_type {m} --model_path {model_name_dict[m]} --result_path result_1215/general_ability --batch_size {batch_size} --do_sample --modified --modify_path result_1215/lvs_select_random_60_{layer_num_dict[m]}/{m} --suffix random --ppid {pid}"
        command_list.append(command)

# %%
print(command_list, flush=True)

# %%
cuda_id_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
cuda_id_list = cuda_id_list.split(",")
cuda_id_list = [id.strip() for id in cuda_id_list]
process_per_device = 1
gpu_semaphores = {cuda_id: Semaphore(process_per_device) for cuda_id in cuda_id_list}

# Additional code to track and clean subprocesses
def run_command(cuda_id, command):
    global running_processes
    with gpu_semaphores[cuda_id]:
        # print(if_end, flush=True)
        if if_end:
            return
        full_command = f"CUDA_VISIBLE_DEVICES=\"{cuda_id}\" " + command
        try:
            process = subprocess.Popen(
                full_command, 
                shell=True,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                # text=True
            )
            running_processes.append(process)
            
            stdout, stderr = process.communicate()
            return_code = process.returncode
            
            if return_code != 0:
                logger.error(f"Process failed with return code {return_code}")
                logger.error(f"Command: {full_command}")
                logger.error(f"Error output: {stderr}")
        except Exception as e:
            logger.error(f"Exception occurred: {str(e)}")
            logger.error(f"Command that caused the exception: {full_command}")
        finally:
            if process in running_processes:
                running_processes.remove(process)

# %%
threads:list[Thread] = []
running_processes:list[subprocess.Popen] = []
if_end = False
for idx, command in enumerate(command_list):
    cuda_id = cuda_id_list[idx % len(cuda_id_list)]
    thread = Thread(target=run_command, args=(cuda_id, command))
    threads.append(thread)
    thread.start()

def cleanup():
    print("Cleanup started...", flush=True)
    global if_end
    if_end = True
    print("Cleaning up...", flush=True)
    # Kill any process running "src/server_answer.py"
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and f"--ppid {pid}" in " ".join(cmdline):
                logger.info(f"Killing process {proc.pid} running: {' '.join(cmdline)}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Terminate all tracked subprocesses
    for process in running_processes:
        try:
            logger.info(f"Terminating subprocess with PID {process.pid}")
            process.terminate()
            process.wait(timeout=5)  # Wait for graceful termination
        except (subprocess.TimeoutExpired, psutil.NoSuchProcess):
            try:
                process.kill()  # Force kill if termination takes too long
            except psutil.NoSuchProcess:
                pass # Process already ended

def sig_handler(signum, frame):
    cleanup()
    sys.exit(0)
atexit.register(cleanup)
signal.signal(signal.SIGTERM, sig_handler)
signal.signal(signal.SIGINT, sig_handler)

for thread in threads:
    thread.join()
