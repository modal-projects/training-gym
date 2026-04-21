"""Tutorial source for `lightning_fabric_demo` — parsed by generate_tutorial.py.

Minimal PyTorch Lightning Fabric multi-node demo, based on
https://lightning.ai/docs/overview/train-models/multi-node-training.
Uses Lightning's Fabric launcher to configure and launch a multi-node
training job via our `lightning` framework.
"""

TUTORIAL_METADATA = {
    'framework': '`lightning`',
    'cluster_shape': '2 × 8×H100',
    'summary': 'Transformer on WikiText2 (Fabric DDP)',
    'order': 70,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # Multi-node Lightning Fabric

    Minimal PyTorch Lightning Fabric multi-node demo, based on the
    [Lightning multi-node training guide](https://lightning.ai/docs/overview/train-models/multi-node-training#lightning-fabric).

    Lightning's `Fabric` launcher handles all the DDP / FSDP / DeepSpeed
    plumbing — we just wrap it with a Modal cluster so it scales across
    nodes.
    """


@notebook_only
@shell("! pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.frameworks.lightning import (
        LightningConfig,
        LightningFrameworkConfig,
    )


@markdown
def _script_section():
    """
    ## Training script

    Tiny Lightning Fabric loop pulled from the upstream docs — a small
    transformer on WikiText2. Fabric handles the rank/world-size/device
    plumbing; the script itself doesn't reference any multi-node specifics.
    """


@code
def _define_script():
    import textwrap

    TRAIN_SCRIPT = textwrap.dedent(r'''
        import lightning as L
        import torch
        import torch.nn.functional as F
        from lightning.pytorch.demos import Transformer, WikiText2
        from torch.utils.data import DataLoader


        def main():
            L.seed_everything(42)

            fabric = L.Fabric()
            fabric.launch()

            with fabric.rank_zero_first():
                dataset = WikiText2(download=True)

            train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
            model = Transformer(vocab_size=dataset.vocab_size)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            model, optimizer = fabric.setup(model, optimizer)
            train_dataloader = fabric.setup_dataloaders(train_dataloader)

            max_steps = 1
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= max_steps:
                    break
                input, target = batch
                output = model(input, target)
                loss = F.nll_loss(output, target.view(-1))
                fabric.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                fabric.print(f"iter {batch_idx}/{max_steps} - loss {loss.item():.4f}")


        if __name__ == "__main__":
            main()
    ''').lstrip("\n")


@markdown
def _config_section():
    """
    ## Experiment config

    `LightningConfig` carries the Fabric knobs (`strategy`, `precision`,
    `accelerator`) + the usual cluster shape. No `dataset`/`model`/`wandb`
    containers needed for this demo — the script downloads WikiText2 on
    its own.
    """


@code
def _define_config():
    lightning_framework_config = LightningFrameworkConfig(
        gpu="H100",
        n_nodes=2,
        gpus_per_node=8,
        train_script_source=TRAIN_SCRIPT,
        accelerator="gpu",
        strategy="ddp",
        precision="bf16-mixed",
    )

    my_training_run = LightningConfig(
        framework_config=lightning_framework_config,
    )


@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = my_training_run.build_app()


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    uv run modal run tutorials/lightning_fabric_demo/lightning_fabric_demo.py::app.download_dataset
    uv run modal run tutorials/lightning_fabric_demo/lightning_fabric_demo.py::app.upload_script
    uv run modal run --detach tutorials/lightning_fabric_demo/lightning_fabric_demo.py::app.train
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and run one stage per cell:
    """


@notebook_only
@code
def _invoke_download_dataset():
    with app.run():
        app.download_dataset.remote()


@notebook_only
@code
def _invoke_upload_script():
    with app.run():
        app.upload_script.remote()


@notebook_only
@code
def _invoke_train():
    with app.run():
        app.train.remote()
