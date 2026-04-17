"""No-op decorators interpreted by generate_tutorial.py.

At runtime these do nothing — the generator AST-walks input files and
extracts cells based on decorator names. Kept as real callables so the
input files are importable in an IDE without red squiggles.
"""


def markdown(fn):
    return fn


def code(fn):
    return fn


def py_only(fn):
    """Restrict a `@markdown` or `@code` cell to the generated .py file only."""
    return fn


def notebook_only(fn):
    """Restrict a `@markdown` or `@code` cell to the generated .ipynb file only."""
    return fn


def shell(command: str):
    """Emit `command` verbatim as a notebook code cell.

    Use when you need shell magic (e.g. `! pip install ...`) or any other
    non-Python content that must appear exactly as-written in the cell. The
    decorated function's body is ignored — use `pass`.
    """

    def wrap(fn):
        return fn

    return wrap
