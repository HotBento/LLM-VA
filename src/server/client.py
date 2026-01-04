import requests
import queue
import threading
import time
import socket
from urllib.parse import urlparse
from loguru import logger

class QwenToxicEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()

        def _worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    break
                if len(item) == 3:
                    inputs, outputs, th = item
                elif len(item) == 2:
                    inputs, th = item
                    outputs = None
                else:
                    inputs = item[0]
                    outputs = None
                    th = 0.5
                try:
                    results = self.evaluate(inputs, outputs, th)
                    self.result_queue.put(results)
                except Exception as e:
                    self.result_queue.put(e)
                finally:
                    self.input_queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def evaluate(self, inputs: list[str], outputs: list[str]=None, th: float = 0.5) -> tuple[list[bool], list[float]]:
        """
        调用服务器端的 QwenToxicEvaluator evaluate 方法。
        :param inputs: 输入列表
        :param th: 阈值
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/toxic"
        payload = {"inputs": inputs, "th": th}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")


class QwenAnswerEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        
        def wait_for_server(host, port, timeout=200):
            start_time = time.time()
            print(f"Waiting for server {host}:{port} to be available...")
            while True:
                try:
                    with socket.create_connection((host, port), timeout=2):
                        break
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Server {host}:{port} not available after {timeout} seconds")
                    time.sleep(1)

        # Parse host and port from server_url
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"Invalid server_url: {server_url}")
        wait_for_server(host, port)

        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()

        def _worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    break
                if len(item) == 3:
                    inputs, outputs, th = item
                elif len(item) == 2:
                    inputs, outputs = item
                    th = 0.5
                try:
                    results = self.evaluate(inputs, outputs, th)
                    self.result_queue.put(results)
                except Exception as e:
                    self.result_queue.put(e)
                finally:
                    self.input_queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def evaluate(self, inputs: list[str], outputs: list[str], th: float = 0.5) -> tuple[list[bool], list[float]]:
        """
        调用服务器端的 QwenAnswerEvaluator evaluate 方法。
        :param inputs: 输入列表
        :param outputs: 输出列表
        :param th: 阈值
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/answer"
        payload = {"inputs": inputs, "outputs": outputs, "th": th}
        response = requests.post(url, json=payload)
        # logger.debug(f"response: {response.json()}")
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")

class SelfAnswerEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        
        def wait_for_server(host, port, timeout=200):
            start_time = time.time()
            print(f"Waiting for server {host}:{port} to be available...")
            while True:
                try:
                    with socket.create_connection((host, port), timeout=2):
                        break
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Server {host}:{port} not available after {timeout} seconds")
                    time.sleep(1)

        # Parse host and port from server_url
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"Invalid server_url: {server_url}")
        wait_for_server(host, port)

        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()

        def _worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    break
                if len(item) == 3:
                    inputs, outputs, th = item
                elif len(item) == 2:
                    inputs, outputs = item
                    th = 0.5
                try:
                    results = self.evaluate(inputs, outputs, th)
                    self.result_queue.put(results)
                except Exception as e:
                    self.result_queue.put(e)
                finally:
                    self.input_queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def evaluate(self, inputs: list[str], outputs: list[str], th: float = 0.5) -> tuple[list[bool], list[float]]:
        """
        调用服务器端的 QwenAnswerEvaluator evaluate 方法。
        :param inputs: 输入列表
        :param outputs: 输出列表
        :param th: 阈值
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/self_answer"
        payload = {"inputs": inputs, "outputs": outputs, "th": th}
        response = requests.post(url, json=payload)
        # logger.debug(f"response: {response.json()}")
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")

class LlamaGuardEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        
        def wait_for_server(host, port, timeout=200):
            start_time = time.time()
            print(f"Waiting for server {host}:{port} to be available...")
            while True:
                try:
                    with socket.create_connection((host, port), timeout=2):
                        break
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Server {host}:{port} not available after {timeout} seconds")
                    time.sleep(1)

        # Parse host and port from server_url
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"Invalid server_url: {server_url}")
        wait_for_server(host, port)

        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()

        def _worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    break
                if len(item) == 2:
                    inputs, outputs = item

                try:
                    results = self.evaluate(inputs, outputs)
                    self.result_queue.put(results)
                except Exception as e:
                    self.result_queue.put(e)
                finally:
                    self.input_queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def evaluate(self, inputs: list[str], outputs: list[str]) -> tuple[list[bool], list[float]]:
        """
        调用服务器端的 LlamaGuardEvaluator evaluate 方法。
        :param inputs: 输入列表
        :param th: 阈值
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/guard"
        payload = {"inputs": inputs, "outputs": outputs}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")

class QwenGuardEvaluatorClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        
        def wait_for_server(host, port, timeout=200):
            start_time = time.time()
            print(f"Waiting for server {host}:{port} to be available...")
            while True:
                try:
                    with socket.create_connection((host, port), timeout=2):
                        break
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Server {host}:{port} not available after {timeout} seconds")
                    time.sleep(1)

        # Parse host and port from server_url
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port
        if host is None or port is None:
            raise ValueError(f"Invalid server_url: {server_url}")
        wait_for_server(host, port)

        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()

        def _worker():
            while True:
                item = self.input_queue.get()
                if item is None:
                    break
                if len(item) == 3:
                    inputs, outputs, eval_type = item

                try:
                    results = self.evaluate(inputs, outputs, eval_type)
                    self.result_queue.put(results)
                except Exception as e:
                    self.result_queue.put(e)
                finally:
                    self.input_queue.task_done()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def evaluate(self, inputs: list[str], outputs: list[str], eval_type: list[str]) -> tuple[list[bool], list[float]]:
        """
        调用服务器端的 QwenGuardEvaluator evaluate 方法。
        :param inputs: 输入列表
        :param outputs: 输出列表
        :param eval_type: 评估类型
        :return: (is_behavior_present, prediction)
        """
        url = f"{self.server_url}/evaluate/guard"
        payload = {"inputs": inputs, "outputs": outputs, "eval_type": eval_type}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["results"], data["predictions"]
        else:
            raise Exception(f"Error from server: {response.status_code}, {response.text}")

def get_server(server_url: str, port: int) -> str:
    """
    获取服务器的 IP 地址。
    :param server_url: 服务器地址
    :param port: 端口号
    :return: IP 地址
    """
    if not server_url.startswith("http://"):
        server_url = "http://" + server_url
    if server_url.endswith("/"):
        server_url = server_url[:-1]
    return f"{server_url}:{port}"

if __name__ == "__main__":
    # 示例服务器地址
    server = "127.0.0.1"
    port = 4397
    SERVER_URL = get_server(server, port)

    # 初始化客户端
    toxic_client = QwenToxicEvaluatorClient(SERVER_URL)
    answer_client = QwenAnswerEvaluatorClient(SERVER_URL)

    # 示例调用 QwenToxicEvaluatorClient
    try:
        inputs = ["This is a test input for toxicity evaluation."]
        th = 0.5
        toxic_results, toxic_predictions = toxic_client.evaluate(inputs, th=th)
        print("Toxic Evaluation Results:", toxic_results)
        print("Toxic Predictions:", toxic_predictions)
    except Exception as e:
        print("Error during toxic evaluation:", e)

    # 示例调用 QwenAnswerEvaluatorClient
    try:
        inputs = ["What is the capital of France?"]
        outputs = ["The capital of France is Paris."]
        th = 0.5
        answer_results, answer_predictions = answer_client.evaluate(inputs, outputs, th)
        print("Answer Evaluation Results:", answer_results)
        print("Answer Predictions:", answer_predictions)
    except Exception as e:
        print("Error during answer evaluation:", e)