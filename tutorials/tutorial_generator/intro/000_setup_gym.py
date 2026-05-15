"""Tutorial source for `000_setup_gym` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "summary": "Install the training-gym package and deploy the observability dashboard to your Modal account.",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
@notebook_only
def _intro():
    """
    # Setup the training-gym

    In this tutorial you'll install `modal-training-gym` and deploy the observability dashboard that tracks your training runs, deployments, and evals.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass

@notebook_only
@shell("! training-gym setup")
def _setup():
    pass

@notebook_only
@markdown
def _setup_results():
    """
    The training-gym dashboard is now deployed to your Modal account — open the URL printed above to see it.

    To deploy to a different Modal environment:
    ```bash
    MODAL_ENVIRONMENT=<environment> training-gym setup
    ```
    """

@notebook_only
@markdown
def _congrats():
    """
    You're all set! The dashboard will automatically discover training runs, deployments, and evals as you work through the next tutorials.
    """