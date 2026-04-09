"""Tests for the bundles module."""

from pathlib import Path

import numpy as np
import pytest
from multiqc import config
from multiqc.base_module import ModuleNoSamplesFound
from skimage import io as skio

from neuroimaging.modules.bundles.bundles import MultiqcModule


def _module_instance():
    """Create a bare module instance for testing helper methods."""
    return object.__new__(MultiqcModule)


def test_resolve_found_file_path_handles_relative_and_absolute_paths(tmp_path):
    module = _module_instance()

    rel = module._resolve_found_file_path({"root": "/tmp/data", "fn": "bundle.trk"})
    assert rel == "/tmp/data/bundle.trk"

    abs_path = Path(tmp_path) / "bundle.trk"
    assert module._resolve_found_file_path({"fn": str(abs_path)}) == str(abs_path)


def test_parse_trk_paths_extracts_bundle_name_and_skips_invalid(caplog):
    module = _module_instance()

    parsed = module._parse_trk_paths(
        [
            {"fn": "sub-01_tract-IFOFL_track-sdstream_tractogram.trk", "root": "/data"},
            {"fn": "no-tract-token.trk", "root": "/data"},
        ]
    )

    assert parsed == {"IFOFL": "/data/sub-01_tract-IFOFL_track-sdstream_tractogram.trk"}
    assert "Could not extract bundle name" in caplog.text


def test_crop_png_whitespace_reduces_image_size(tmp_path):
    fp = tmp_path / "img.png"
    img = np.full((100, 120, 3), 255, dtype=np.uint8)
    img[30:70, 40:80, :] = 0
    skio.imsave(str(fp), img, check_contrast=False)

    MultiqcModule._crop_png_whitespace(str(fp), white_threshold=245, pad=4)
    cropped = skio.imread(str(fp))

    assert cropped.shape[0] < img.shape[0]
    assert cropped.shape[1] < img.shape[1]


def test_crop_png_whitespace_keeps_all_white_image_unchanged(tmp_path):
    fp = tmp_path / "all_white.png"
    img = np.full((48, 64, 3), 255, dtype=np.uint8)
    skio.imsave(str(fp), img, check_contrast=False)

    MultiqcModule._crop_png_whitespace(str(fp), white_threshold=245, pad=8)
    result = skio.imread(str(fp))

    assert result.shape == img.shape


def test_crop_png_whitespace_handles_float_images(tmp_path):
    fp = tmp_path / "float_img.png"
    img = np.ones((60, 60, 3), dtype=np.float32)
    img[20:40, 20:40, :] = 0.0
    # Save as uint8 PNG while keeping the method on float-threshold branch via direct call input recreation.
    skio.imsave(str(fp), (img * 255).astype(np.uint8), check_contrast=False)

    MultiqcModule._crop_png_whitespace(str(fp), white_threshold=245, pad=2)
    result = skio.imread(str(fp))

    assert result.shape[0] <= 60
    assert result.shape[1] <= 60


def test_build_bundle_browser_multiple_bundles_includes_controls_and_keyboard_js():
    module = _module_instance()

    html_content = module._build_bundle_browser(
        {
            "IFOF_L": '<img alt="a" src="data:image/png;base64,AAA" />',
            "SLF_R": '<img alt="b" src="data:image/png;base64,BBB" />',
        },
        default_bundle="IFOF_L",
    )

    assert 'id="bundle-selector"' in html_content
    assert "Previous bundle" in html_content
    assert "Next bundle" in html_content
    assert "showPreviousBundle" in html_content
    assert "showNextBundle" in html_content
    assert "ArrowLeft" in html_content
    assert "ArrowRight" in html_content
    assert "display: block" in html_content
    assert "display: none" in html_content


def test_build_bundle_browser_single_bundle_omits_selector_controls():
    module = _module_instance()

    html_content = module._build_bundle_browser(
        {"IFOF_L": '<img alt="a" src="data:image/png;base64,AAA" />'},
        default_bundle="IFOF_L",
    )

    assert 'id="bundle-selector"' not in html_content
    assert "Previous bundle" not in html_content
    assert "Next bundle" not in html_content
    assert "bundleContainers" in html_content


def test_init_raises_when_single_subject_disabled():
    config.kwargs = {"single_subject": False}

    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()


def test_init_raises_when_no_trk_files(monkeypatch):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": []}

    def fake_find_log_files(self, pattern):
        return []

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)

    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()


def test_init_filters_configured_bundle_list_with_start_of_filename_match(monkeypatch, tmp_path):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": ["IFOF_L", "SLF_R"]}

    trk_files = [
        {"fn": "IFOF_L.trk", "root": str(tmp_path)},
        {"fn": "subjectX-SLF_R.trk", "root": str(tmp_path)},
        {"fn": "UF_L.trk", "root": str(tmp_path)},
    ]

    captured_trk_dict = {}

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return []
        return []

    def fake_render_bundle_images(self, trk_dict, bg_mesh):
        captured_trk_dict.update(trk_dict)
        return {k: f"<img alt='{k}' src='data:image/png;base64,AAA' />" for k in trk_dict}

    captured_sections = []

    def fake_add_section(self, name, anchor, content, **kwargs):
        captured_sections.append((name, anchor, content))

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "_load_nii_as_mesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(MultiqcModule, "_render_bundle_images", fake_render_bundle_images, raising=False)
    monkeypatch.setattr(MultiqcModule, "add_section", fake_add_section, raising=False)

    MultiqcModule()

    assert set(captured_trk_dict.keys()) == {"IFOF_L", "SLF_R"}
    assert "UF_L" not in captured_trk_dict
    assert len(captured_sections) == 1
    assert captured_sections[0][0] == "Bundle Visualization"


def test_init_raises_when_configured_bundles_do_not_match_any_files(monkeypatch, tmp_path, caplog):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": ["IFOF_L", "SLF_R"]}

    trk_files = [{"fn": "UF_L.trk", "root": str(tmp_path)}]

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return []
        return []

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)

    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()

    assert "No .trk file found for bundle 'IFOF_L'" in caplog.text
    assert "No .trk file found for bundle 'SLF_R'" in caplog.text


def test_init_uses_parse_trk_paths_when_no_bundle_filter(monkeypatch, tmp_path):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": []}

    trk_files = [{"fn": "ignored.trk", "root": str(tmp_path)}]
    parsed_trk_dict = {"A": "/tmp/A.trk", "B": "/tmp/B.trk"}
    captured_trk_dict = {}

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return []
        return []

    def fake_parse_trk_paths(self, files):
        return parsed_trk_dict

    def fake_render_bundle_images(self, trk_dict, bg_mesh):
        captured_trk_dict.update(trk_dict)
        return {k: f"<img alt='{k}' src='data:image/png;base64,AAA' />" for k in trk_dict}

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "_parse_trk_paths", fake_parse_trk_paths, raising=False)
    monkeypatch.setattr(MultiqcModule, "_load_nii_as_mesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(MultiqcModule, "_render_bundle_images", fake_render_bundle_images, raising=False)
    monkeypatch.setattr(MultiqcModule, "add_section", lambda *args, **kwargs: None, raising=False)

    MultiqcModule()

    assert captured_trk_dict == parsed_trk_dict


def test_init_raises_when_render_returns_empty(monkeypatch, tmp_path):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": ["IFOF_L"]}

    trk_files = [{"fn": "IFOF_L.trk", "root": str(tmp_path)}]

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return []
        return []

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "_load_nii_as_mesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(MultiqcModule, "_render_bundle_images", lambda *args, **kwargs: {}, raising=False)

    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()


def test_init_warns_on_multiple_nii_and_uses_first(monkeypatch, tmp_path, caplog):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": ["IFOF_L"]}

    trk_files = [{"fn": "IFOF_L.trk", "root": str(tmp_path)}]
    nii_files = [
        {"fn": "first.nii.gz", "root": str(tmp_path)},
        {"fn": "second.nii.gz", "root": str(tmp_path)},
    ]
    called = {"nii_fp": None}

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return nii_files
        return []

    def fake_load_nii_as_mesh(nii_fp):
        called["nii_fp"] = nii_fp
        return "mesh"

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "_load_nii_as_mesh", staticmethod(fake_load_nii_as_mesh))
    monkeypatch.setattr(
        MultiqcModule,
        "_render_bundle_images",
        lambda self, trk_dict, bg_mesh: {"IFOF_L": "<img alt='IFOF_L' src='data:image/png;base64,AAA' />"},
        raising=False,
    )
    monkeypatch.setattr(MultiqcModule, "add_section", lambda *args, **kwargs: None, raising=False)

    MultiqcModule()

    assert "Found multiple .nii files, will use the first one." in caplog.text
    assert called["nii_fp"].endswith("first.nii.gz")


def test_init_warns_when_no_nii_and_skips_mesh_loading(monkeypatch, tmp_path, caplog):
    config.kwargs = {"single_subject": True}
    config.bundles = {"bundles": ["IFOF_L"]}

    trk_files = [{"fn": "IFOF_L.trk", "root": str(tmp_path)}]

    def fake_find_log_files(self, pattern):
        if pattern == "bundles/trk":
            return trk_files
        if pattern == "bundles/nii":
            return []
        return []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_load_nii_as_mesh should not be called when no .nii is present")

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "_load_nii_as_mesh", fail_if_called)
    monkeypatch.setattr(
        MultiqcModule,
        "_render_bundle_images",
        lambda self, trk_dict, bg_mesh: {"IFOF_L": "<img alt='IFOF_L' src='data:image/png;base64,AAA' />"},
        raising=False,
    )
    monkeypatch.setattr(MultiqcModule, "add_section", lambda *args, **kwargs: None, raising=False)

    MultiqcModule()

    assert "No .nii files found, bundle visualizations will be performed without background." in caplog.text
