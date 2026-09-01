from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

import click
from pydantic import BaseModel

from modal_training_gym.common.models.base import HFModelConfiguration
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe
from scripts.api_reference_manifest import SDK_SIDEBAR_CLASSES


def _load_generator():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    generate_api_reference = importlib.import_module("generate_api_reference")
    manifest = importlib.import_module("api_reference_manifest")

    return generate_api_reference, manifest.API_REFERENCE_MANIFEST


def _render(class_name: str) -> str:
    generate, manifest = _load_generator()
    entry = next(e for e in manifest if e["class_name"] == class_name)
    mod = importlib.import_module(entry["module"])
    obj = getattr(mod, entry["class_name"])
    return generate.generate_class_page(obj, entry, 0)


def _preamble(page: str) -> str:
    marker = "\n## "
    idx = page.find(marker)
    return page[: idx if idx != -1 else len(page)]


def test_manifest_entries_resolve_and_fill_index_groups() -> None:
    generate, manifest = _load_generator()
    index = generate.generate_index_page(manifest)
    assert index.startswith("---\norder: 0\n---\n\n# SDK reference\n")

    links: set[str] = set()
    for entry in manifest:
        obj = getattr(importlib.import_module(entry["module"]), entry["class_name"])
        assert obj is not None
        link = generate._sdk_link(entry)
        assert link not in links
        links.add(link)

    for group, heading in (
        ("models", "Models"),
        ("datasets", "Datasets"),
        ("recipes", "Recipes"),
        ("training", "Training"),
        ("deployment", "Deployment"),
    ):
        expected = {
            entry["sidebar_label"] for entry in manifest if entry["group"] == group
        }
        assert set(_index_table_names(index, heading)) == expected


def test_datasetconfig_lede_does_not_repeat_field_docs() -> None:
    page = _render("DatasetConfig")
    preamble = _preamble(page)
    assert "output_format : str" not in preamble
    assert "Describes *what* the data is" not in preamble
    assert "**Attributes**" in page
    assert "## Parameters" not in page
    assert 'class="tg-param"' in page
    assert "<strong>dataset_id</strong> <code>str</code>" in page
    assert 'Default: ""' in page
    assert "split: Literal['all', 'train', 'eval']" in page
    assert 'split: "Literal' not in page
    assert "## `load`" in page
    assert "## `load(" not in page
    assert "## Methods" not in page
    assert "**Source:**" not in page
    assert "| `output_format` |" not in page


def test_trainresult_uses_google_field_docs_and_skips_constructor() -> None:
    page = _render("TrainResult")
    assert "## Constructor" not in page
    assert "<factory>" not in page
    assert "<strong>app_name</strong>" in page
    assert "<strong>model_config</strong> <code>ModelConfig | None</code>" in page
    assert ":meth:`load`" not in page
    assert "## `load`" in page
    assert "## `checkpoints`" in page
    assert "**Returns**" in page
    assert "model: ModelConfig" in page
    assert "model(" not in page


def test_modelconfig_skips_empty_kwargs_constructor() -> None:
    page = _render("ModelConfig")
    assert "## Constructor" not in page
    assert "**kwargs" not in page
    assert ":class:`ParsedResponse`" not in page
    assert "<strong>response_parser</strong>" not in page
    assert "<strong>model_name</strong>" not in page
    assert "## `parse_response`" in page


def test_inherited_class_omits_dedicated_inheritance_prose() -> None:
    page = _render("Qwen3_4B")
    assert "from modal_training_gym.common.models.qwen3_4b import Qwen3_4B" in page
    assert "**Inherits from:**" not in page


def test_kwargs_model_preset_documents_public_overrides() -> None:
    page = _render("Qwen3_4B")
    assert "### response_parser" not in page
    assert "parse_qwen3_response" in page
    assert "ModelArchitecture(...)" in page
    assert "num_layers=36" not in page
    assert "<function parse_qwen3_response" not in page
    assert "<strong>model_name</strong> <code>str</code>" in page
    assert "<strong>architecture</strong> <code>ModelArchitecture | None</code>" in page
    assert "<strong>response_parser</strong> <code>Callable" in page
    assert "<strong>model_path</strong>" not in page

    recipe_page = _render("Qwen3_5_4B_Miles_Recipe")
    assert "## `model_config_class`" not in recipe_page


def test_customdeployment_matches_config_template() -> None:
    page = _render("CustomDeployment")
    assert "Modal Endpoint" not in page
    assert "## Constructor" not in page
    assert "**Attributes**" in page
    assert "## Parameters" not in page
    assert "## Fields" not in page
    assert "<strong>unauthenticated</strong> <code>bool</code>" in page
    assert "Default: True" in page
    assert "## `chat`" in page
    assert "## `chat(" not in page
    assert "chat(self" not in page
    assert "## Methods" not in page
    assert "**Source:**" not in page
    assert "**Returns**" in page


def test_sglang_recipe_docs_only_reference_current_fields() -> None:
    generate, _ = _load_generator()
    sections = generate._parse_docstring_sections(SglangRecipe.__doc__ or "")
    field_names = {field.name for field in dataclasses.fields(SglangRecipe)}

    assert set(sections.params) <= field_names
    assert {
        "num_gpus",
        "tp_size",
        "dp_size",
        "image_run_commands",
        "image_env",
    }.isdisjoint(sections.params)


def test_cli_pages_use_modal_lists_not_tables() -> None:
    generate, _ = _load_generator()
    pages = generate.collect_cli_pages()
    run = next(page for page in pages if page.slug == "run")
    page = generate.generate_cli_page(run, 1)
    index = generate.generate_cli_index(pages)
    assert "## Options" not in page
    assert "## Commands" not in page
    assert "## Usage" not in page
    assert "| Name | Description |" not in page
    assert "| Name | Description |" in index
    assert "**Usage**:" in page
    assert "**Options**:" in page
    assert "**Commands**:" in page
    assert "* [`training-gym run get`](#training-gym-run-get):" in page
    assert "## `training-gym run get`" in page
    head = page.split("\n## ", 1)[0]
    assert "--help" in head
    assert "| [`training-gym run`](/reference/cli/run) |" in index
    assert "* [`training-gym run`](/reference/cli/run):" not in index
    assert "**Examples**:" in page
    assert "[1<=x<=20000]" not in page
    assert "[default:" not in page
    for cli_page in generate.collect_cli_pages():
        if not cli_page.children:
            continue
        group_page = generate.generate_cli_page(cli_page, 1)
        group_head = group_page.split("\n## ", 1)[0]
        assert group_head.index("**Usage**:") < group_head.index("**Options**:")
        assert group_head.index("**Options**:") < group_head.index("**Commands**:")


def test_cli_pages_preserve_multiline_help_and_epilogs() -> None:
    generate, _ = _load_generator()
    command = click.Command(
        "sample",
        help="Short  summary.\n\nRequired usage constraint.",
        epilog="Examples:\n  training-gym sample --force",
        params=[
            click.Option(
                ["--count"],
                default=3,
                help="Number of samples.",
                show_default=True,
                type=click.IntRange(1, 9),
            ),
            click.Option(["--out"], help="Output path.", required=True),
        ],
    )
    page = generate.CliPage(
        label="training-gym sample",
        slug="sample",
        panel="Commands",
        summary=generate._cli_help_parts(command)[0],
        command=command,
    )
    rendered = generate.generate_cli_page(page, 1)

    assert rendered.count("Short summary.") == 1
    assert "Required usage constraint." in rendered
    assert rendered.index("Required usage constraint.") < rendered.index("**Usage**:")
    assert "**Examples**:" in rendered
    assert "training-gym sample --force" in rendered
    assert "Default `3`; range `1` to `9`." in rendered
    assert "Output path. Required." in rendered


def test_every_cli_page_keeps_source_help_details_and_epilogs() -> None:
    generate, _ = _load_generator()
    for page in generate.collect_cli_pages():
        rendered = generate.generate_cli_page(page, 1)
        normalized_rendered = " ".join(rendered.split())
        for command_page in (page, *page.children):
            help_text = inspect.cleandoc(command_page.command.help or "")
            paragraphs = [
                " ".join(paragraph.split())
                for paragraph in re.split(r"\n\s*\n", help_text)
                if paragraph.strip()
            ]
            for paragraph in paragraphs[1:]:
                assert paragraph in normalized_rendered
            epilog = inspect.cleandoc(command_page.command.epilog or "")
            if epilog:
                for line in epilog.removeprefix("Examples:\n").splitlines():
                    assert line.strip() in rendered


def test_family_parsers_are_not_standalone_pages() -> None:
    generate, manifest = _load_generator()
    names = {entry["class_name"] for entry in manifest}
    for parser in (
        "parse_qwen3_response",
        "parse_qwen3_6_response",
        "parse_glm_response",
    ):
        assert parser not in names
    assert not any(entry["class_type"] == "function" for entry in manifest)
    page = _render("ModelConfig")
    assert "## `parse_response`" in page
    assert "Parse model text with `response_parser`." in page
    glm = _render("GLM_4_7")
    assert "Parse model text with `response_parser`." in glm
    assert "Qwen3 think blocks" not in glm


def test_recipe_summaries_match_configured_cluster_and_colocation() -> None:
    generate, manifest = _load_generator()

    for entry in manifest:
        if entry["group"] != "recipes":
            continue
        recipe_class = getattr(
            importlib.import_module(entry["module"]), entry["class_name"]
        )
        recipe = recipe_class()
        summary = generate._class_lede(recipe_class)
        allocation = recipe.gpu_allocation
        if allocation.colocate:
            expected = (
                f"{allocation.total_nodes} "
                f"{'node' if allocation.total_nodes == 1 else 'nodes'} with "
                f"{allocation.gpus_per_node} {recipe.gpu_type} GPUs"
            )
            if allocation.total_nodes > 1:
                expected += " each"
        else:
            trainer_nodes = allocation.actor_gpus // allocation.gpus_per_node
            rollout_nodes = allocation.rollout_gpus // allocation.gpus_per_node
            expected = (
                f"{trainer_nodes} trainer nodes and {rollout_nodes} rollout nodes, "
                f"each with {allocation.gpus_per_node} {recipe.gpu_type} GPUs"
            )
        assert expected in summary, entry["class_name"]


def test_class_lede_does_not_inherit_parent_essay() -> None:
    generate, _ = _load_generator()

    class _Bare(HFModelConfiguration):
        pass

    assert generate._class_lede(_Bare) == ""
    assert "snapshot_download" in (inspect.getdoc(_Bare) or "")
    assert "snapshot_download" not in _preamble(_render("Qwen3_ASR_1_7B"))


def test_endpoint_returns_are_distinct() -> None:
    page = _render("Endpoint")
    endpoint_docs_url = "https://modal.com/docs/guide/endpoints"
    assert f"[Modal Endpoint]({endpoint_docs_url})" in _preamble(page)
    assert page.count(endpoint_docs_url) == 1
    launch_section = page.split("## `launch`", 1)[1].split("\n## ", 1)[0]
    assert "**Returns**" in launch_section
    assert "## `wait_until_ready`" in page
    wait_section = page.split("## `wait_until_ready`", 1)[1].split("\n## ", 1)[0]
    assert "**Returns**" not in wait_section


def test_reference_docstrings_preserve_structured_sections() -> None:
    generate, _ = _load_generator()
    sections = generate._parse_docstring_sections(
        """Keep this. Keep this detail.

        Args:
            value: Keep both sentences. This matters.
        Yields:
            One streamed value.
        Raises:
            ValueError: The value is invalid.
        """
    )
    assert sections.description == "Keep this. Keep this detail."
    assert sections.params == {"value": "Keep both sentences. This matters."}
    assert sections.yields == "One streamed value."
    assert sections.returns == ""
    assert sections.raises == (
        generate._RaiseDoc("ValueError", "The value is invalid."),
    )
    assert generate._first_sentence("E.g. keep this. Drop this.") == "E.g. keep this."
    assert (
        generate._first_sentence("Deploy to U.S. regions. Drop this.")
        == "Deploy to U.S. regions."
    )
    assert (
        generate._first_sentence("Supports A, B, etc. Drop this.")
        == "Supports A, B, etc."
    )
    rendered = "\n".join(generate._structured_section_lines(sections))
    assert rendered.index("**Yields**") < rendered.index("**Raises**")
    assert "**Returns**" not in rendered

    prose = generate._parse_docstring_sections(
        """Arguments win because defaults are applied before them::

            run()

        A vision run requires its own reward: the text scorer does not handle images.
        """
    )
    assert prose.attributes == {}
    assert "run()" in prose.description
    assert "requires its own reward" in prose.description


def test_behavior_pages_separate_attributes_from_constructor_parameters() -> None:
    generate, _ = _load_generator()

    class Behavior:
        """Behavioral handle.

        Attributes:
            state: Current state.
        """

        state: str

        def __init__(
            self,
            documented: str,
            undocumented: str | None = None,
            *extras: str,
        ):
            """Create the handle.

            Parameters:
                documented: Public constructor input.
                extras: Additional constructor inputs.
            """

    entry = {
        "class_name": "Behavior",
        "sidebar_label": "Behavior",
        "module": "example",
    }
    page = generate.generate_class_page(Behavior, entry, 0)
    constructor = page.split("## Constructor", 1)[1]

    assert "**Attributes**" in page
    assert page.index("**Attributes**") < page.index("## Constructor")
    assert "Current state." in page
    assert "**Parameters**" in constructor
    assert "Public constructor input." in constructor
    assert "<strong>*extras</strong>" in constructor
    assert "<strong>undocumented</strong>" not in constructor


def test_documented_property_attribute_has_no_descriptor_default() -> None:
    page = _render("ModalRayCluster")
    assert "<strong>is_head</strong> <code>bool</code>" in page
    assert "property object" not in page
    client = page.split("## `client`", 1)[1].split("\n## ", 1)[0]
    assert "client(self)" not in client
    assert "**Raises**" in client


def test_annotation_fallback_keeps_subclass_overrides() -> None:
    generate, _ = _load_generator()

    class Base:
        value: int

    class Child(Base):
        value: str

    Child.__annotations__["value"] = "MissingType"

    type_hint, _ = generate._get_class_attrs(Child)["value"]
    assert type_hint == "MissingType"


def test_structural_types_render_all_fields_including_undocumented_ones() -> None:
    generate, _ = _load_generator()

    @dataclasses.dataclass
    class SampleDataclass:
        described: str
        undocumented: int

    class SampleModel(BaseModel):
        described: str
        undocumented: int

    class SamplePayload(TypedDict):
        """Structured request payload.

        Args:
            required: Required value.
            optional: Optional value.
        """

        required: str
        optional: NotRequired[int]

    class PartialPayload(TypedDict, total=False):
        optional: int

    entry = {
        "class_name": "SamplePayload",
        "sidebar_label": "SamplePayload",
        "module": __name__,
        "class_type": "behavior",
        "group": "models",
    }
    typed_dict_page = generate.generate_class_page(SamplePayload, entry, 0)
    assert "**Attributes**" in typed_dict_page
    assert "<strong>required</strong> <code>str</code>" in typed_dict_page
    assert "<strong>optional</strong> <code>NotRequired[int]</code>" in typed_dict_page
    assert "Required value." in typed_dict_page
    assert "## `fromkeys`" not in typed_dict_page

    for cls in (SampleDataclass, SampleModel):
        page = generate.generate_class_page(
            cls,
            {**entry, "class_name": cls.__name__, "sidebar_label": cls.__name__},
            0,
        )
        assert "<strong>described</strong> <code>str</code>" in page
        assert "<strong>undocumented</strong> <code>int</code>" in page
        undocumented_card = page.split("<strong>undocumented</strong>", 1)[1].split(
            "</div>", 1
        )[0]
        assert "<p>" not in undocumented_card

    optional_type, _ = generate._get_class_attrs(PartialPayload)["optional"]
    assert generate._format_type(optional_type) == "NotRequired[int]"


def test_manifest_label_does_not_make_ordinary_fields_structural() -> None:
    generate, _ = _load_generator()

    class Ordinary:
        """Ordinary behavior.

        Attributes:
            documented: Public state.
        """

        documented: str
        undocumented: str

    page = generate.generate_class_page(
        Ordinary,
        {
            "class_name": "Ordinary",
            "sidebar_label": "Ordinary",
            "module": __name__,
            "class_type": "config_data",
            "group": "models",
        },
        0,
    )
    assert "<strong>documented</strong>" in page
    assert "<strong>undocumented</strong>" not in page


def _index_table_names(index: str, group_heading: str) -> list[str]:
    section = index.split(f"## {group_heading}", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"\| \[`([^`]+)`\]", section)


def test_sidebar_matches_overview_tables() -> None:
    generate, manifest = _load_generator()

    sidebar = generate.build_reference_sidebar()
    index = generate.generate_index_page(manifest)
    cli_pages = generate.collect_cli_pages()
    cli_index = generate.generate_cli_index(cli_pages)

    sidebar_labels = [item["label"] for item in sidebar["sdk"]]
    assert sidebar["sdk"] == sorted(
        [
            {
                "label": entry["sidebar_label"],
                "link": f"/reference/{entry['class_name'].lower()}",
            }
            for entry in manifest
            if entry["class_name"] in SDK_SIDEBAR_CLASSES
        ],
        key=lambda item: item["label"].casefold(),
    )
    assert all("items" not in item for item in sidebar["sdk"])
    assert sidebar_labels == sorted(sidebar_labels, key=str.casefold)
    assert not any(
        label in {"Models", "Datasets", "Recipes", "Training", "Deployment"}
        for label in sidebar_labels
    )
    overview_names = {
        name
        for heading in ("Models", "Datasets", "Recipes", "Training", "Deployment")
        for name in _index_table_names(index, heading)
    }
    assert set(sidebar_labels) <= overview_names

    sidebar_classes = {
        entry["class_name"]
        for entry in manifest
        if entry["class_name"] in SDK_SIDEBAR_CLASSES
    }
    assert sidebar_classes == SDK_SIDEBAR_CLASSES

    cli_table_names = re.findall(r"\| \[`([^`]+)`\]", cli_index)
    assert sidebar["cli"] == [
        {"label": page.label, "link": f"/reference/cli/{page.slug}"}
        for page in cli_pages
    ]
    assert [item["label"] for item in sidebar["cli"]] == cli_table_names
    assert all(item["link"] != "/reference/cli" for item in sidebar["cli"])
    assert "training-gym run" in cli_table_names
    assert "training-gym setup" in cli_table_names


def test_params_follow_modal_default_and_docs_heuristic() -> None:
    generate, manifest = _load_generator()

    pages = [_render(entry["class_name"]) for entry in manifest]
    pages.append(generate.generate_cli_index(generate.collect_cli_pages()))
    for cli_page in generate.collect_cli_pages():
        pages.append(generate.generate_cli_page(cli_page, 1))

    empty_detail = re.compile(
        r'<div class="tg-param">\s*<p>[^<]*(?:<code>[^<]*</code>)?</p>\s*<p>\s*</p>',
        re.DOTALL,
    )
    for page in pages:
        assert "Default: None" not in page
        assert "Default: &lt;lambda&gt;" not in page
        assert empty_detail.search(page) is None

    for entry in manifest:
        obj = getattr(importlib.import_module(entry["module"]), entry["class_name"])
        if generate._uses_structural_field_rendering(obj):
            continue
        page = generate.generate_class_page(obj, entry, 0)
        for card in re.findall(
            r'<div class="tg-param">(.*?)</div>', page, flags=re.DOTALL
        ):
            assert card.count("<p>") == 2, entry["class_name"]

    qwen_vl = _render("Qwen3_VL_8B_Recipe")
    assert "Default: ['vision_model']" in qwen_vl
    for class_name in ("Qwen3_ASR_1_7B_Recipe", "Gemma4_26B_A4B_Recipe"):
        page = _render(class_name)
        assert "<strong>image_run_commands</strong>" in page
        image_commands = page.split("<strong>image_run_commands</strong>", 1)[1].split(
            "</div>", 1
        )[0]
        assert "Default:" not in image_commands
