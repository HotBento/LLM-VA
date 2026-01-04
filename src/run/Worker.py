import subprocess
from threading import Thread
import os
from loguru import logger
class Worker():
    def __init__(self, cuda_id, command_queue, running_processes, command_list, if_end):
        self.cuda_id = cuda_id
        self.command_queue = command_queue
        self.running_processes = running_processes
        self.command_list = command_list
        self.if_end = if_end
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        while not self.command_queue.empty():
            if self.if_end:
                break
            command = self.command_queue.get()
            full_command = f"CUDA_VISIBLE_DEVICES=\"{self.cuda_id}\" " + command
            logger.info(f"Running command on GPU {self.cuda_id}: {command}")
            try:
                process = subprocess.run(full_command, shell=True)
                self.running_processes.append(process)
                return_code = process.returncode
                if return_code != 0:
                    logger.error(f"Command failed with return code {return_code} on GPU {self.cuda_id}: {command}")
                    self.command_list.append(command)
            except Exception as e:
                logger.error(f"Exception occurred while running command on GPU {self.cuda_id}: {command}. Exception: {e}")
                self.command_list.append(command)
            finally:
                if process in self.running_processes:
                    self.running_processes.remove(process)
    def end(self):
        self.thread.join(0.1)
