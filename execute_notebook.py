"""Execute this trusted notebook in-process with IPython and capture its outputs.

Useful in environments where a Jupyter kernel cannot open local sockets. This
executes every Python cell, including assertions, and preserves rich PNG output.
For ordinary interactive work, open the notebook in Jupyter instead.
"""
from pathlib import Path
import os
import sys
import nbformat

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter_runtime" / "ipython"))
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support

shell = InteractiveShell.instance()
configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
path = ROOT / "vsa_gestalt_closure.ipynb"
notebook = nbformat.read(path, as_version=4)
count = 0
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        continue
    count += 1
    print(f"Executing code cell {count} (notebook cell {index})", flush=True)
    cell.outputs = []
    cell.execution_count = count
    with capture_output() as captured:
        result = shell.run_cell(cell.source, store_history=False)
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output(
            "display_data", data=output.data, metadata=output.metadata))
    error = result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output(
            "error", ename=type(error).__name__, evalue=str(error), traceback=[captured.stdout]))
        nbformat.write(notebook, path)
        print(captured.stdout, file=sys.stderr)
        raise RuntimeError(f"Notebook cell {index} failed") from error
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print(f"Successfully executed {count} code cells.", flush=True)
print((ROOT / "outputs" / "run_summary.json").read_text())
