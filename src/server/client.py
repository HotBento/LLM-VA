import requests
import queue
import threading
import time
import socket
from urllib.parse import urlparse


class QwenGuardEvaluatorClient:
    def __init__(self, server_url: str):
        """Initialize the QwenGuard evaluator client.

        Parameters
        ----------
        server_url : str
            URL of the server hosting the evaluation service.
        """
        self.server_url = server_url
        
        def wait_for_server(host, port, timeout=200):
            """Wait for server to become available.

            Parameters
            ----------
            host : str
                Server hostname or IP address.
            port : int
                Server port number.
            timeout : int, optional
                Maximum seconds to wait for server availability.
            """
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
        """Call the server-side QwenGuardEvaluator evaluate method.

        Parameters
        ----------
        inputs : list[str]
            List of input texts to evaluate.
        outputs : list[str]
            List of output texts to evaluate.
        eval_type : list[str]
            Evaluation type specifying the guard criteria.

        Returns
        -------
        tuple[list[bool], list[float]]
            Tuple of (is_behavior_present, prediction_scores).
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
    """Construct full server address with protocol and port.

    Parameters
    ----------
    server_url : str
        Server address or domain name.
    port : int
        Server port number.

    Returns
    -------
    str
        Full server URL in format http://host:port.
    """
    if not server_url.startswith("http://"):
        server_url = "http://" + server_url
    if server_url.endswith("/"):
        server_url = server_url[:-1]
    return f"{server_url}:{port}"
