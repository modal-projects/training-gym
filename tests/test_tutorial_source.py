from __future__ import annotations

from tutorials.swe_bench.main import agentic_swe_image, build_training_config


class _Image:
    def __init__(self) -> None:
        self.package_sources: list[tuple[str, bool]] = []

    def uv_pip_install(self, *packages):
        return self

    def add_local_python_source(self, package: str, *, copy: bool):
        self.package_sources.append((package, copy))
        return self


def test_swe_overlay_includes_tutorial_package_source() -> None:
    image = _Image()

    result = agentic_swe_image(image)

    assert result is image
    assert image.package_sources == [("tutorials.swe_bench", True)]


def test_swe_config_uses_worker_import_path() -> None:
    config = build_training_config()

    assert config.recipe.extra_config["custom_generate_function_path"] == (
        "tutorials.swe_bench.generate.generate"
    )
