import pytest

from modal_training_gym.frameworks.miles.launcher import (
    _is_complete_torch_dist_checkpoint,
)
from modal_training_gym.common.run import has_torch_dist_checkpoint


@pytest.mark.parametrize("legacy_common", [False, True])
def test_completed_conversion_with_legacy_or_embedded_common_state(
    tmp_path, legacy_common
):
    release = tmp_path / "release"
    release.mkdir()
    for name in [
        ".metadata",
        "__0_0.distcp",
        *(["common.pt"] if legacy_common else []),
    ]:
        (release / name).touch()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    assert has_torch_dist_checkpoint(
        str(tmp_path), is_complete=_is_complete_torch_dist_checkpoint
    )


@pytest.mark.parametrize(
    "names", [[], ["__0_0.distcp"], ["__0_0.distcp", "common.pt"], [".metadata"]]
)
def test_incomplete_conversion_is_not_a_cache_hit(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()
    assert not _is_complete_torch_dist_checkpoint(str(tmp_path))
