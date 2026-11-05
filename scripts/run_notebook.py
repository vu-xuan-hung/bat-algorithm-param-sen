#!/usr/bin/env python
"""Execute trusted plain-Python notebook cells and save genuine rich outputs.

This is a sequential Python runner, not a Jupyter kernel or a magic-command interpreter.
It needs only the existing research dependencies. Existing output files are refused.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image


def rich_output(value: Any) -> dict[str, Any]:
    """Serialize actual tables, PNG images or text as a notebook display output."""
    data: dict[str, Any] = {"text/plain": [repr(value)]}
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    elif hasattr(value, "_repr_html_"):
        html = value._repr_html_()
        if html is not None:
            data["text/html"] = [html]
    return {"output_type": "display_data", "data": data, "metadata": {}}


def _capture_display(stream: io.StringIO, outputs: list[dict[str, Any]]) -> tuple[Any, Any]:
    def flush_stream() -> None:
        text = stream.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": [text]})
            stream.seek(0)
            stream.truncate()

    def display(*values: Any) -> None:
        flush_stream()
        outputs.extend(rich_output(value) for value in values)

    return flush_stream, display


def execute_notebook(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Execute cells in one namespace; preserve errors and refuse output replacement."""
    notebook = json.loads(input_path.read_text())
    if notebook.get("nbformat") != 4:
        raise ValueError("Only nbformat 4 notebooks are supported.")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    # Reserve the destination before any experiment work.
    with output_path.open("x") as destination:
        namespace: dict[str, Any] = {"__name__": "__main__"}
        failure: BaseException | None = None
        count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            outputs: list[dict[str, Any]] = []
            stream = io.StringIO()

            flush_stream, display = _capture_display(stream, outputs)

            namespace["display"] = display
            cell["execution_count"] = count
            cell["outputs"] = outputs
            source = "".join(cell["source"])
            try:
                tree = ast.parse(source, filename=f"{input_path}:cell-{count}")
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    exec(compile(tree, f"{input_path}:cell-{count}", "exec"), namespace)
            except BaseException as error:
                failure = error
                flush_stream()
                outputs.append(
                    {
                        "output_type": "error",
                        "ename": type(error).__name__,
                        "evalue": str(error),
                        "traceback": traceback.format_exc().splitlines(),
                    }
                )
                break
            flush_stream()
        notebook["metadata"]["execution_verification"] = {
            "runner": "sequential Python cells; not a Jupyter kernel",
            "python": sys.version.split()[0],
            "executed_code_cells": count,
            "status": "failed" if failure else "passed",
        }
        json.dump(notebook, destination, ensure_ascii=False, indent=1, allow_nan=False)
        destination.write("\n")
    if failure:
        raise RuntimeError(f"Notebook failed; traceback saved to {output_path}") from failure
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ba-matplotlib")
    execute_notebook(args.input.resolve(), args.output.resolve())
    print(args.output)


if __name__ == "__main__":
    main()
