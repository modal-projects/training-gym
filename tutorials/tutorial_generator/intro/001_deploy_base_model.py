"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare",
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
    # Deploy the base model

    This tutorial shows you how to deploy a base model on Modal to use for evaluation. Here, we will use Qwen3-4B as the base model, but
    you can use any other model that is supported by the training-gym, or configure your own.
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

    To deploy the base model, we can use the `DeploymentConfig` class. This allows us to deploy the model on Modal and get a handle to the deployed model.
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
    Now that the model has come alive, we can request it to generate text.

    Note that this is creating an SGLang server app and starting it up, then sending a request, so the first request should take a few minutes to complete.
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
    The `chat_template_kwargs` parameter allows us to pass in a dictionary of kwargs to the chat template.
    This is useful for passing in additional parameters to the chat template, such as the enable_thinking parameter.
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
@code
def _look_at_dashboard():
    """
    Remember the training gym dashboard you deployed in the previous tutorial?
    We can now see the base model deployment in the dashboard.
    """

@notebook_only
@code
def _look_at_dashboard_code():
    print(base_model_deployment.modal_app_url)