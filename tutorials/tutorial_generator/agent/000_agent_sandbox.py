# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `000_agent_sandbox` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "Modal Sandbox",
    "cluster_shape": "1 × 1×H100",
    "summary": "Build an LLM agent harness with a self-hosted model and Modal Sandbox tool execution",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "CustomDeployment",
        "Qwen3_5_9B",
        "SglangRecipe",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Build an agent harness with a self-hosted model + Modal Sandboxes

    This tutorial builds an LLM agent loop from scratch using a
    **self-hosted model** served on Modal. The agent can use two
    tools — **list a directory** and **read a file** — and every
    tool call executes inside a
    [Modal Sandbox](https://modal.com/docs/guide/sandbox), an
    isolated container with its own filesystem.

    What you'll learn:
    1. Deploy a model with `CustomDeployment` and get an
       OpenAI-compatible endpoint.
    2. Use the OpenAI Python SDK pointed at your self-hosted
       endpoint (no API key needed).
    3. Create a Modal Sandbox with files pre-loaded via
       `filesystem.write_text`.
    4. Define tools that run shell commands inside the sandbox
       with `sandbox.exec`.
    5. Wire everything into an agent loop with tool calling.

    The entire stack runs on Modal — model serving, tool execution,
    and the sandbox — so you control cost, latency, and data privacy.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run --with openai \\
      python tutorials/agent/000_agent_sandbox/000_agent_sandbox.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main\n"
    "if importlib.util.find_spec('openai') is None:\n"
    "    %uv pip install -q openai"
)
def _install():
    pass


@code
def _imports():
    import json

    import modal
    import openai

    from modal_training_gym import (
        CustomDeployment,
        Qwen3_5_9B,
    )
    from modal_training_gym.deploy_recipes import SglangRecipe


@markdown
def _tools_section():
    """
    ## Define the tools

    We define two tools and a dispatcher function. Each tool runs a
    command inside a sandbox via `sandbox.exec` and captures
    stdout/stderr. The tool definitions follow the OpenAI
    function-calling schema.

    The dispatcher takes the sandbox as an argument so it can be
    called from the agent loop after the sandbox is created.
    """


@code
def _define_tools():
    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": (
                    "List the contents of a directory. Returns one entry per "
                    "line. Directories have a trailing slash."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the directory.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the full contents of a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    def dispatch_tool(sb, name: str, arguments: str) -> str:
        args = json.loads(arguments)
        if name == "list_directory":
            proc = sb.exec("ls", "-1F", args["path"])
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            return stdout if proc.returncode == 0 else f"Error: {stderr}"

        elif name == "read_file":
            proc = sb.exec("cat", args["path"])
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            return stdout if proc.returncode == 0 else f"Error: {stderr}"

        return f"Unknown tool: {name}"


@markdown
def _deploy_section():
    """
    ## Deploy the model

    `CustomDeployment.launch()` launches an sglang-backed inference
    server on Modal and returns a `CustomDeployment` with a live URL.
    The server exposes an **OpenAI-compatible** `/v1/chat/completions`
    endpoint, so we point the standard OpenAI Python SDK at it.

    We pass `extra_server_args={"--tool-call-parser": "qwen3_coder",
    "--reasoning-parser": "qwen3"}` to the `SglangRecipe` so the server
    parses Qwen3.5's XML-style tool-call format and strips any inline
    thinking blocks before returning structured `tool_calls`.
    Without this, the model emits tool calls as raw text.
    """


@code
def _deploy_model():
    recipe = SglangRecipe(
        extra_server_args={
            "--tool-call-parser": "qwen3_coder",
            "--reasoning-parser": "qwen3",
        },
    )
    deployment = CustomDeployment.launch(
        Qwen3_5_9B(),
        recipe=recipe,
        unauthenticated=True,
    )
    deployment.wait_until_ready()
    print(f"Model URL: {deployment.url}")

    client = openai.OpenAI(
        base_url=f"{deployment.url}/v1",
        api_key="not-needed",
    )


@markdown
def _sandbox_section():
    """
    ## Create a sandbox with sample files

    We spin up a long-lived Sandbox running `sleep infinity` so it
    stays alive while the agent issues commands. After creation we
    write a small project tree into the sandbox's filesystem.
    """


@code
def _create_sandbox():
    sandbox_app = modal.App.lookup("agent-sandbox-tutorial", create_if_missing=True)

    sandbox = modal.Sandbox._experimental_create(
        "sleep", "infinity",
        app=sandbox_app,
        image=modal.Image.debian_slim(python_version="3.12"),
        timeout=600,
    )

    FILES = {
        "/repo/README.md": (
            "# My Project\n\n"
            "A small Python utility that computes Fibonacci numbers.\n\n"
            "## Usage\n\n"
            "```bash\n"
            "python fib.py 10\n"
            "```\n"
        ),
        "/repo/fib.py": (
            "import sys\n\n\n"
            "def fibonacci(n: int) -> list[int]:\n"
            '    """Return the first n Fibonacci numbers."""\n'
            "    if n <= 0:\n"
            "        return []\n"
            "    seq = [0, 1]\n"
            "    while len(seq) < n:\n"
            "        seq.append(seq[-1] + seq[-2])\n"
            "    return seq[:n]\n\n\n"
            'if __name__ == "__main__":\n'
            "    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10\n"
            "    print(fibonacci(count))\n"
        ),
        "/repo/tests/test_fib.py": (
            "from fib import fibonacci\n\n\n"
            "def test_empty():\n"
            "    assert fibonacci(0) == []\n\n\n"
            "def test_one():\n"
            "    assert fibonacci(1) == [0]\n\n\n"
            "def test_ten():\n"
            "    result = fibonacci(10)\n"
            "    assert len(result) == 10\n"
            "    assert result[-1] == 34\n"
        ),
        "/repo/pyproject.toml": (
            '[project]\nname = "fib"\nversion = "0.1.0"\n'
            'requires-python = ">=3.12"\n'
        ),
    }

    for path, content in FILES.items():
        sandbox.filesystem.write_text(content, path)

    print(f"Sandbox created: {sandbox.object_id}")


@markdown
def _agent_loop_section():
    """
    ## The agent loop

    The loop uses the OpenAI SDK's tool-calling protocol:
    1. Send messages + tool definitions to the self-hosted model.
    2. If the model returns `tool_calls`, execute each one in the
       sandbox and append the results as `tool` messages.
    3. Repeat until the model produces a final text response.

    We cap iterations at 10 to avoid runaway loops. We also pass
    `enable_thinking=False` in `chat_template_kwargs` so Qwen3.5
    skips its internal chain-of-thought block and responds
    directly — this keeps tool-call parsing clean.

    That's only a chat-template hint, though, and `--reasoning-parser
    qwen3` routes any thinking that does slip through into
    `reasoning_content` instead of `content`. We read `content` first
    and fall back to `reasoning_content` so a thinking-only turn still
    prints an answer.
    """


@code
def _agent_loop():
    MODEL = deployment.served_model_name
    MAX_ITERATIONS = 10

    messages = [
        {
            "role": "user",
            "content": (
                "Explore the /repo directory. List what files exist, read "
                "each one, and give me a summary of the project — what it "
                "does, how the code is structured, and whether the tests "
                "look correct."
            ),
        },
    ]

    print("Starting agent loop...\n")

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            # `--reasoning-parser qwen3` splits any <think> block out of
            # `content` and into `reasoning_content`. `enable_thinking=False`
            # above should keep thinking off entirely, but that's a chat-template
            # hint the server is free to ignore — fall back so a thinking-only
            # turn still prints something instead of `None`.
            final = choice.message.content or getattr(
                choice.message, "reasoning_content", None
            )
            print(f"Agent response:\n{final}")
            break

        messages.append(choice.message)

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                print(f"  [{i+1}] Calling {tc.function.name}({tc.function.arguments})")
                result = dispatch_tool(sandbox, tc.function.name, tc.function.arguments)
                print(f"       → {len(result)} chars returned")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

    else:
        print("Reached max iterations without a final response.")


@markdown
def _cleanup_section():
    """
    ## Clean up

    Terminate the sandbox so it doesn't keep running (and billing)
    after we're done.
    """


@code
def _cleanup():
    sandbox.terminate()
    print("Sandbox terminated.")


@markdown
def _next_steps():
    """
    ## Next steps

    This tutorial showed how to combine a self-hosted model with
    sandbox tool execution — no external API keys required. The
    model runs on Modal, the tools run on Modal, and everything
    is under your control.

    Ideas to extend this:
    - **Add a `run_command` tool** so the agent can execute
      arbitrary shell commands (run tests, install packages).
    - **Add a `write_file` tool** using
      `sandbox.filesystem.write_text` so the agent can modify
      code.
    - **Swap models** — try `Qwen3_6_27B` for harder tasks, or
      `Qwen3_5_4B` for lower cost.
    - **Snapshot the filesystem** with
      `sandbox.snapshot_filesystem()` to create a reusable
      `modal.Image` from the sandbox state.
    """
