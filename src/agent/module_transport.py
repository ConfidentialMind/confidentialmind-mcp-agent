import os
import sys
from pathlib import Path
from typing import List, Optional

from fastmcp.client.transports import StdioTransport


class ModuleStdioTransport(StdioTransport):
    """Transport for running Python modules with proper package context."""

    def __init__(
        self,
        module_path: str,
        args: Optional[List[str]] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
        python_cmd: str = sys.executable,
    ):
        """
        Initialize transport to run a Python module.

        Args:
            module_path: Dotted path to the Python module (e.g., 'src.tools.postgres_mcp')
            args: Additional arguments to pass to the module
            env: Environment variables to set for the subprocess
            cwd: Current working directory for the subprocess
            python_cmd: Python command to use (default: sys.executable)
        """
        # Build command to run module with -m flag
        module_args = ["-m", module_path]
        if args:
            module_args.extend(args)
        else:
            # If no args provided, add a default arg to signal stdio mode
            module_args.append("--stdio")

        if env is None:
            env = {}

        env_copy = os.environ.copy()
        env_copy["FastMCP_TRANSPORT"] = "stdio"
        env_copy.update(env)

        super().__init__(command=python_cmd, args=module_args, env=env_copy, cwd=cwd)
        self.module_path = module_path

    def __repr__(self) -> str:
        return f"<ModuleStdioTransport(module='{self.module_path}')>"


def path_to_module_path(file_path: str) -> str:
    """
    Convert a file path to a Python module path.

    Args:
        file_path: Path to a Python file

    Returns:
        Module path (e.g., 'src.tools.postgres_mcp' for '/path/to/src/tools/postgres_mcp/__main__.py')
    """
    path = Path(file_path).resolve()

    # If the path is to __main__.py, use its parent directory
    if path.name == "__main__.py" or path.name == "main.py":
        path = path.parent
    # If the path is to a .py file, remove the extension
    elif path.suffix == ".py":
        path = path.with_suffix("")

    # Convert path to module notation
    parts = []
    current = path

    # Walk up the directory tree until we find a directory that doesn't contain __init__.py
    while (current / "__init__.py").exists() or (current.parent / "__init__.py").exists():
        parts.insert(0, current.name)
        current = current.parent

        # Safety check to prevent infinite loops
        if current == current.parent:
            break

    return ".".join(parts)
