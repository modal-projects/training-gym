"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "summary": "Setup the training-gym",
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

    This tutorial shows you how to setup the training-gym.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass

@notebook_only
@shell("training-gym setup")
def _setup():
    pass

@notebook_only
@markdown
def _setup_results():
    """
    The training-gym dashboard is now deployed to your Modal account.
    You can access it at the URL printed above.

    If you want to deploy it to a different environment, you can run the following command:
    ```bash
    MODAL_ENVIRONMENT=<environment> training-gym setup
    ```
    """

@notebook_only
@markdown
def _congrats():
    """
    Congratulations! You have successfully setup the training-gym.
    """