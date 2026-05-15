"""Tutorial source for `001_deploy_base_model` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Deploy a model to a Modal inference endpoint and send it your first generation request.",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_4B",
        "DeploymentConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Deploy a base model

    Before training or evaluating, you need a running inference endpoint. In this tutorial you'll deploy Qwen3-4B to Modal using `DeploymentConfig` and send it a generation request. Any model in the training-gym catalog works — Qwen3-4B is just a convenient default.
    """

@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/intro/001_deploy_base_model.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass

@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        Qwen3_4B,
    )


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    `DeploymentConfig.serve()` launches an SGLang inference server on Modal and returns a deployment handle you can call directly.
    """


@code
def _serve_base_model():
    base_model = Qwen3_4B()
    base_model_deployment = DeploymentConfig(
        model=base_model,
    ).serve()
    print(f"Base model deployed to {base_model_deployment.url}")

@notebook_only
@markdown
def _qualitative_eval_of_base_model():
    """
    The first call cold-starts the SGLang server, so expect it to take a few minutes. Subsequent requests are fast.
    """

@notebook_only
@code
def _qualitative_eval_of_base_model_code():
    response = base_model_deployment.generate(
        "Write a short story about a cat.",
    )
    print(response)

@markdown
def _chat_template_kwargs():
    """
    ## Disabling thinking mode

    Qwen3 defaults to "thinking" mode, where the model reasons internally before answering. Pass `enable_thinking=False` via `chat_template_kwargs` to get a direct response instead.
    """

@notebook_only
@code
def _qualitative_eval_of_base_model_code():
    response = base_model_deployment.generate(
        "Write a short story about a cat.",
        chat_template_kwargs={"enable_thinking": False},
    )
    print(response)


@notebook_only
@markdown
def _look_at_dashboard():
    """
    ## Check the dashboard

    Your deployment should now be visible in the training-gym dashboard you set up earlier.
    """

@notebook_only
@code
def _look_at_dashboard_code():
    print(base_model_deployment.modal_app_url)