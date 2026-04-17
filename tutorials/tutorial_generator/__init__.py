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
