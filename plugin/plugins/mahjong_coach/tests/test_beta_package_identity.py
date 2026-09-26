from scripts.build_mahjong_coach_beta import _artifact_stem, _release_plugin_id


def test_beta_releases_keep_one_upgrade_identity() -> None:
    assert _release_plugin_id(version="0.3.27b1", release="20260813") == "mahjong_coach_beta"
    assert _release_plugin_id(version="0.3.27b6", release="20260814") == "mahjong_coach_beta"


def test_beta_artifacts_remain_version_distinguishable() -> None:
    assert _artifact_stem(version="0.3.27b1") == "mahjong_coach_beta_0_3_27b1"
    assert _artifact_stem(version="0.3.27b6") == "mahjong_coach_beta_0_3_27b6"
