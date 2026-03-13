"""Tests for the connectivity module."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
from multiqc import config, report
from multiqc.base_module import ModuleNoSamplesFound


def _binary_matrix(non_zero_count, shape=(10, 10)):
    """Create a matrix with a fixed number of non-zero entries."""
    matrix = np.zeros(shape[0] * shape[1], dtype=float)
    matrix[:non_zero_count] = 1.0
    return matrix.reshape(shape)


def _write_lut(path, size=10):
    """Write a minimal LUT file for heatmap labels."""
    with open(path, "w") as handle:
        json.dump({str(index + 1): f"Region {index + 1}" for index in range(size)}, handle)


def _register_connectivity_files(data_dir, matrix_files, lut_file="atlas_LUT.json"):
    """Populate MultiQC report file metadata for connectivity inputs."""
    config.analysis_dir = [data_dir]
    report.files["connectivity/matrices"] = [
        {
            "fn": filename,
            "root": data_dir,
            "s_name": os.path.splitext(filename)[0],
            "sp_key": "connectivity/matrices",
        }
        for filename in matrix_files
    ]
    report.files["connectivity/lut"] = [
        {
            "fn": lut_file,
            "root": data_dir,
            "s_name": os.path.splitext(lut_file)[0],
            "sp_key": "connectivity/lut",
        }
    ]


@pytest.fixture
def reset_multiqc():
    """Reset MultiQC state before each test."""
    config.reset()
    report.reset()
    if "connectivity/matrices" not in config.sp:
        config.update_dict(config.sp, {"connectivity/matrices": {"fn": "*stat-*.npy"}})
    if "connectivity/lut" not in config.sp:
        config.update_dict(config.sp, {"connectivity/lut": {"fn": "*LUT.json"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with connectivity matrices and a LUT."""
    tmpdir = tempfile.mkdtemp()
    density_cases = {
        "sub-WARN_stat-fa.npy": 24,
        "sub-PASS1_stat-fa.npy": 25,
        "sub-PASS2_stat-fa.npy": 26,
        "sub-PASS3_stat-fa.npy": 27,
        "sub-FAIL_stat-fa.npy": 100,
    }

    _write_lut(os.path.join(tmpdir, "atlas_LUT.json"))
    for filename, non_zero_count in density_cases.items():
        np.save(os.path.join(tmpdir, filename), _binary_matrix(non_zero_count))

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the connectivity module can be imported."""
    from neuroimaging.modules.connectivity import connectivity

    assert hasattr(connectivity, "MultiqcModule")


def test_parse_connectivity_file_extracts_bids_sample_name(reset_multiqc):
    """Test parsing a connectivity file with a BIDS-style sample name."""
    from neuroimaging.modules.connectivity import connectivity

    tmpdir = tempfile.mkdtemp()
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    filename = "sub-P001_ses-BL_run-01_stat-fa.npy"

    try:
        np.save(os.path.join(tmpdir, filename), matrix)
        module = object.__new__(connectivity.MultiqcModule)
        parsed = module.parse_connectivity_file(
            {
                "fn": filename,
                "root": tmpdir,
                "s_name": "ignored",
                "sp_key": "connectivity/matrices",
            }
        )

        assert parsed["sample_name"] == "sub-P001_ses-BL_run-01"
        np.testing.assert_array_equal(parsed["values"], matrix)
    finally:
        shutil.rmtree(tmpdir)


def test_parse_connectivity_file_falls_back_to_s_name(reset_multiqc):
    """Test parsing falls back to MultiQC metadata when no BIDS ID is present."""
    from neuroimaging.modules.connectivity import connectivity

    tmpdir = tempfile.mkdtemp()
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    filename = "atlas_stat-fa.npy"

    try:
        np.save(os.path.join(tmpdir, filename), matrix)
        module = object.__new__(connectivity.MultiqcModule)
        parsed = module.parse_connectivity_file(
            {
                "fn": filename,
                "root": tmpdir,
                "s_name": "subject_from_metadata",
                "sp_key": "connectivity/matrices",
            }
        )

        assert parsed["sample_name"] == "subject_from_metadata"
        np.testing.assert_array_equal(parsed["values"], matrix)
    finally:
        shutil.rmtree(tmpdir)


def test_compute_density(reset_multiqc):
    """Test density calculation on a simple connectivity matrix."""
    from neuroimaging.modules.connectivity import connectivity

    module = object.__new__(connectivity.MultiqcModule)
    matrix = np.array([[1.0, 0.0], [1.0, 1.0]])

    assert module.compute_density(matrix) == pytest.approx(0.75)


def test_frequency_matrix(reset_multiqc):
    """Test frequency matrix calculation across subjects."""
    from neuroimaging.modules.connectivity import connectivity

    matrices = {
        "sub-A": {"fa": np.array([[1.0, 0.0], [0.0, 1.0]])},
        "sub-B": {"md": np.array([[0.0, 1.0], [1.0, 1.0]])},
    }

    module = object.__new__(connectivity.MultiqcModule)
    frequency = module.frequency_matrix(matrices)

    expected = np.array([[0.5, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(frequency, expected)


def test_global_mode_adds_density_stats_and_statuses(reset_multiqc, test_data_dir):
    """Test cohort mode density statistics and IQR-based status assignment."""
    from neuroimaging.modules.connectivity import connectivity

    config.kwargs = {"single_subject": False}
    matrix_files = [
        "sub-WARN_stat-fa.npy",
        "sub-PASS1_stat-fa.npy",
        "sub-PASS2_stat-fa.npy",
        "sub-PASS3_stat-fa.npy",
        "sub-FAIL_stat-fa.npy",
    ]
    _register_connectivity_files(test_data_dir, matrix_files)

    module = connectivity.MultiqcModule()

    assert module is not None
    assert len(module.sections) == 1
    assert module.sections[0].name == "Connectivity Frequency Matrix"

    section = module.sections[0]
    assert '"sub-WARN": "warn"' in section.status_bar_html
    assert '"sub-PASS2": "pass"' in section.status_bar_html
    assert '"sub-FAIL": "fail"' in section.status_bar_html

    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]
    assert general_stats["sub-WARN"][0].data["density"] == pytest.approx(0.24)
    assert general_stats["sub-PASS2"][0].data["density"] == pytest.approx(0.26)
    assert general_stats["sub-FAIL"][0].data["density"] == pytest.approx(1.0)


def test_configurable_iqr_multiplier_changes_outlier_status(reset_multiqc, test_data_dir):
    """Test that a custom IQR multiplier affects cohort status thresholds."""
    from neuroimaging.modules.connectivity import connectivity

    config.connectivity = {"iqr_multiplier": 100}
    config.kwargs = {"single_subject": False}
    matrix_files = [
        "sub-WARN_stat-fa.npy",
        "sub-PASS1_stat-fa.npy",
        "sub-PASS2_stat-fa.npy",
        "sub-PASS3_stat-fa.npy",
        "sub-FAIL_stat-fa.npy",
    ]
    _register_connectivity_files(test_data_dir, matrix_files)

    module = connectivity.MultiqcModule()
    section = module.sections[0]

    assert '"sub-FAIL": "warn"' in section.status_bar_html
    assert '"sub-PASS2": "pass"' in section.status_bar_html


def test_ignore_samples_filters_general_stats(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimaging.modules.connectivity import connectivity

    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-WARN"]
    matrix_files = [
        "sub-WARN_stat-fa.npy",
        "sub-PASS1_stat-fa.npy",
        "sub-PASS2_stat-fa.npy",
        "sub-PASS3_stat-fa.npy",
        "sub-FAIL_stat-fa.npy",
    ]
    _register_connectivity_files(test_data_dir, matrix_files)

    connectivity.MultiqcModule()

    general_stats = list(report.general_stats_data.values())[0]
    assert "sub-WARN" not in general_stats
    assert "sub-PASS1" in general_stats
    assert len(general_stats) == 4

    config.sample_names_ignore = []


def test_single_subject_mode_adds_metric_selector_and_single_matrix_section(reset_multiqc):
    """Test that single-subject mode uses one selector and one matrix section."""
    from neuroimaging.modules.connectivity import connectivity

    tmpdir = tempfile.mkdtemp()

    try:
        _write_lut(os.path.join(tmpdir, "atlas_LUT.json"), size=3)
        np.save(os.path.join(tmpdir, "sub-SINGLE_stat-fa.npy"), _binary_matrix(3, shape=(3, 3)))
        np.save(os.path.join(tmpdir, "sub-SINGLE_stat-md.npy"), _binary_matrix(5, shape=(3, 3)))

        config.kwargs = {"single_subject": True}
        _register_connectivity_files(
            tmpdir,
            ["sub-SINGLE_stat-fa.npy", "sub-SINGLE_stat-md.npy"],
        )

        module = connectivity.MultiqcModule()

        assert len(module.sections) == 2

        selector_section, matrix_section = module.sections
        assert selector_section.name == "Connectivity Metric Selection"
        assert "<select" in selector_section.content
        assert "FA" in selector_section.content
        assert "MD" in selector_section.content

        assert matrix_section.name == "Connectivity Matrix"
        assert "sub-SINGLE_fa_connectivity_matrix_container" in matrix_section.content
        assert "sub-SINGLE_md_connectivity_matrix_container" in matrix_section.content
        assert "renderConnectivityMetric_sub_SINGLE" in matrix_section.content
    finally:
        shutil.rmtree(tmpdir)


def test_empty_input_raises_module_no_samples_found(reset_multiqc):
    """Test handling of missing connectivity matrix inputs."""
    from neuroimaging.modules.connectivity import connectivity

    tmpdir = tempfile.mkdtemp()

    try:
        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}
        report.files["connectivity/matrices"] = []
        report.files["connectivity/lut"] = []

        with pytest.raises(ModuleNoSamplesFound):
            connectivity.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
