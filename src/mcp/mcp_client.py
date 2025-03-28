# mcp_client.py
import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Dict, Optional

from src.mcp.mcp_protocol import JsonRpcRequest, JsonRpcResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [MCP Client] %(levelname)s: %(message)s"
)


class MCPClient:
    def __init__(self, server_command: str, server_env: Optional[Dict[str, str]] = None):
        self.server_command = server_command.split()  # Split command string into list
        self.server_env = {**os.environ, **(server_env or {})}  # Merge env vars
        self._process: Optional[subprocess.Popen] = None
        self._request_id_counter = 0
        self._response_queue = queue.Queue()
        self._listener_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.start_server()

    def start_server(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                logging.warning("Server process already running.")
                return

            logging.info(f"Starting MCP server: {' '.join(self.server_command)}")
            try:
                self._process = subprocess.Popen(
                    self.server_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,  # Capture stderr for debugging
                    text=True,
                    bufsize=1,  # Line buffered
                    encoding="utf-8",
                    env=self.server_env,
                )
                # Give server a moment to start
                time.sleep(1)
                if self._process.poll() is not None:
                    stderr_output = self._process.stderr.read() if self._process.stderr else "N/A"
                    raise RuntimeError(
                        f"Server process failed to start. Exit code: {self._process.returncode}. Stderr:\n{stderr_output}"
                    )

                self._listener_thread = threading.Thread(
                    target=self._listen_for_responses, daemon=True
                )
                self._listener_thread.start()
                logging.info(f"MCP server process started (PID: {self._process.pid})")

                # Start a thread to monitor stderr
                self._stderr_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
                self._stderr_thread.start()

            except Exception as e:
                logging.error(f"Failed to start MCP server: {e}", exc_info=True)
                self._process = None
                raise

    def _monitor_stderr(self):
        try:
            if self._process and self._process.stderr:
                for line in iter(self._process.stderr.readline, ""):
                    logging.warning(f"[Server STDERR] {line.strip()}")
            logging.info("Server stderr monitoring ended.")
        except Exception as e:
            logging.error(f"Error monitoring server stderr: {e}")

    def _listen_for_responses(self):
        try:
            if self._process and self._process.stdout:
                for line in iter(self._process.stdout.readline, ""):
                    logging.debug(f"Received from server stdout: {line.strip()}")
                    try:
                        response_data = json.loads(line)
                        self._response_queue.put(response_data)
                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON from server: {line.strip()}")
            logging.info("Server response listener thread finished.")
        except Exception as e:
            logging.error(f"Error in server response listener thread: {e}")

    def stop_server(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                logging.info(f"Stopping MCP server process (PID: {self._process.pid})...")
                try:
                    self._process.terminate()  # Try graceful termination
                    self._process.wait(timeout=5)  # Wait a bit
                except subprocess.TimeoutExpired:
                    logging.warning("Server did not terminate gracefully, killing.")
                    self._process.kill()
                except Exception as e:
                    logging.error(f"Error stopping server process: {e}")
                self._process = None
                logging.info("MCP server process stopped.")
            # Wait for listener threads to potentially finish
            if self._listener_thread and self._listener_thread.is_alive():
                # Cannot forcefully join daemon thread easily, just log
                logging.debug("Listener thread is daemon, will exit with main process.")
            if self._stderr_thread and self._stderr_thread.is_alive():
                logging.debug("Stderr thread is daemon, will exit with main process.")

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        with self._lock:
            if not self._process or self._process.poll() is not None:
                logging.error("Server process is not running. Attempting restart.")
                self.start_server()  # Attempt restart
                if not self._process or self._process.poll() is not None:
                    raise RuntimeError(
                        "MCP Server process is not running and could not be restarted."
                    )

            self._request_id_counter += 1
            request_id = self._request_id_counter
            request = JsonRpcRequest(id=request_id, method=method, params=params)

            request_json = request.model_dump_json(exclude_none=True) + "\n"
            logging.debug(f"Sending to server stdin: {request_json.strip()}")

            try:
                if self._process.stdin:
                    self._process.stdin.write(request_json)
                    self._process.stdin.flush()
                else:
                    raise IOError("Server stdin is not available.")
            except (IOError, BrokenPipeError) as e:
                logging.error(
                    f"Failed to send request to server: {e}. Server might have crashed.",
                    exc_info=True,
                )
                self.stop_server()  # Mark server as stopped
                raise RuntimeError(f"Failed to communicate with MCP server: {e}")

        # Wait for the response with the matching ID
        timeout = 30  # seconds timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response_data = self._response_queue.get(timeout=1)
                if response_data.get("id") == request_id:
                    response = JsonRpcResponse.model_validate(response_data)
                    logging.debug(f"Received matching response for ID {request_id}")
                    if response.error:
                        logging.error(f"MCP Server returned error: {response.error}")
                        # Raise an exception or return error structure based on need
                        raise RuntimeError(
                            f"MCP Error: {response.error.get('message', 'Unknown error')}"
                        )
                    return response.result
            except queue.Empty:
                continue  # Keep waiting if queue is empty but timeout not reached
            except Exception as e:
                logging.error(f"Error processing response from queue: {e}")
                raise  # Re-raise validation or other processing errors

        logging.error(f"Timeout waiting for response for request ID {request_id}")
        raise TimeoutError(f"Timeout waiting for MCP server response (ID: {request_id})")

    # --- Convenience methods for MCP calls ---

    def list_resources(self) -> Dict[str, Any]:
        return self._send_request("mcp_listResources")

    def read_resource(self, uri: str) -> Dict[str, Any]:
        return self._send_request("mcp_readResource", {"uri": uri})

    def list_tools(self) -> Dict[str, Any]:
        return self._send_request("mcp_listTools")

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self._send_request("mcp_callTool", {"name": name, "arguments": arguments})

    def __del__(self):
        # Ensure server is stopped when client is garbage collected
        self.stop_server()
