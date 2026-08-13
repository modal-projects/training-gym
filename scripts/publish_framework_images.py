"""Publish the slime and Miles registry images as Modal named Images.

The launchers build from ``Image.from_name(...)`` (``SLIME_IMAGE`` in the slime
launcher, ``MilesRecipe.docker_image``), so the names below must be published
before the first training launch — and republished whenever the pinned
registry references change. Resolving a name never pulls from the registry or
triggers a rebuild, so launches keep using the last published image.

Usage:

    uv run scripts/publish_framework_images.py                    # both frameworks
    uv run scripts/publish_framework_images.py --framework slime
    uv run scripts/publish_framework_images.py --framework miles

Names containing ``/`` are published under an environment prefix
(``environment/name:tag``); the environment must already exist. Publishing
into a *public* environment (usable across all workspaces) additionally
requires a Modal admin identity and a globally deployed underlying image —
that path is gated server-side and not exposed here.
"""

import argparse

import modal

from modal_training_gym.frameworks.slime.launcher import (
    SLIME_IMAGE,
    SLIME_REGISTRY_IMAGE,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

# framework -> (registry source, published named-Image ref)
PUBLISHES: dict[str, tuple[str, str]] = {
    "slime": (SLIME_REGISTRY_IMAGE, SLIME_IMAGE),
    # For Miles the registry tag and the published name are the same string.
    "miles": (MilesRecipe.docker_image, MilesRecipe.docker_image),
}


def publish(framework: str) -> None:
    registry_ref, name = PUBLISHES[framework]
    image = modal.Image.from_registry(registry_ref)
    app = modal.App.lookup("training-gym-image-builds", create_if_missing=True)
    with modal.enable_output():
        built = image.build(app)
    built.publish(name)
    print(f"published {name} (from {registry_ref})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["slime", "miles", "all"], default="all")
    args = parser.parse_args()

    frameworks = ["slime", "miles"] if args.framework == "all" else [args.framework]
    for framework in frameworks:
        publish(framework)


if __name__ == "__main__":
    main()
