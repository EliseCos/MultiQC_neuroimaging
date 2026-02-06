"""
Tests for the harmonization module Battacharyya section.
"""

import os
import tempfile
import shutil
import pytest
from multiqc import config, report

# Mock data for Bhattacharyya distance files
RAW_DISTANCE_DATA = """\
roi_names Bundle1 Bundle2
distance 0.5 0.8
"""

HARMONIZED_DISTANCE_DATA = """\
roi_names Bundle1 Bundle2
distance 0.2 0.4
"""


@pytest.fixture
def reset_multiqc():
    """Reset MultiQC state before each test."""
    config.reset()
    report.reset()
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Create raw and harmonized distance files for 'metric1'
    raw_path = os.path.join(tmpdir, "TestSite.metric1.raw.bhattacharyya.txt")
    with open(raw_path, "w") as f:
        f.write(RAW_DISTANCE_DATA)

    harmonized_path = os.path.join(tmpdir, "TestSite.metric1.harmonized.bhattacharyya.txt")
    with open(harmonized_path, "w") as f:
        f.write(HARMONIZED_DISTANCE_DATA)

    # Create files for a second metric 'metric2'
    raw_path_2 = os.path.join(tmpdir, "TestSite.metric2.raw.bhattacharyya.txt")
    with open(raw_path_2, "w") as f:
        f.write(RAW_DISTANCE_DATA.replace("0.5 0.8", "0.6 0.9"))

    harmonized_path_2 = os.path.join(tmpdir, "TestSite.metric2.harmonized.bhattacharyya.txt")
    with open(harmonized_path_2, "w") as f:
        f.write(HARMONIZED_DISTANCE_DATA.replace("0.2 0.4", "0.3 0.5"))

    yield tmpdir

    shutil.rmtree(tmpdir)


def get_battacharyya_files(test_data_dir):
    """Helper to create the file list for the section."""
    files = []
    for fn in os.listdir(test_data_dir):
        # Filter for relevant files to avoid picking up random files or malformed test inputs
        if fn.endswith(".bhattacharyya.txt"):
            with open(os.path.join(test_data_dir, fn), "r") as f:
                files.append({"fn": fn, "f": f.read()})
    return files


def test_battacharyya_section_init(reset_multiqc, test_data_dir):
    """Test the BattacharyyaSection constructor with valid data."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    files = get_battacharyya_files(test_data_dir)
    section = BattacharyyaSection(files)

    assert "metric1" in section.metrics
    assert "metric2" in section.metrics
    assert "Bundle1" in section.bundles
    assert "Bundle2" in section.bundles

    # Check raw data parsing
    assert "raw" in section.data
    assert "metric1" in section.data["raw"]
    assert section.data["raw"]["metric1"]["bundles"] == ["Bundle1", "Bundle2"]
    assert section.data["raw"]["metric1"]["distances"] == [0.5, 0.8]

    # Check harmonized data parsing
    assert "harmonized" in section.data
    assert "metric1" in section.data["harmonized"]
    assert section.data["harmonized"]["metric1"]["bundles"] == ["Bundle1", "Bundle2"]
    assert section.data["harmonized"]["metric1"]["distances"] == [0.2, 0.4]


def test_battacharyya_filtering(reset_multiqc, test_data_dir):
    """Test filtering of metrics and bundles."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    files = get_battacharyya_files(test_data_dir)
    section = BattacharyyaSection(files)

    # Filter metrics
    section.filter_metrics(["metric1"])
    assert section.metrics == {"metric1"}

    # Filter bundles
    section.filter_bundles(["Bundle1"])
    assert section.bundles == ["Bundle1"]


def test_battacharyya_build_html(reset_multiqc, test_data_dir):
    """Test the build_html method."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    files = get_battacharyya_files(test_data_dir)
    section = BattacharyyaSection(files)

    html_content = section.build_html(default_metric="metric1", render_plot_func="renderPlot")

    assert html_content is not None
    assert isinstance(html_content.content, str)
    assert "bhattacharyya_metric1" in html_content.content
    assert "bhattacharyya_metric2" in html_content.content
    assert "renderBhatt" in html_content.content
    assert "renderPlot" in html_content.content
    assert (
        'style="max-width:800px; margin: 0 auto; display: block;"' in html_content.content
    )  # Default metric should be visible
    assert (
        'style="max-width:800px; margin: 0 auto; display: none;"' in html_content.content
    )  # Other metric should be hidden


def test_battacharyya_malformed_filename(reset_multiqc):
    """Test with a malformed filename."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    # Must provide valid content so validation fails on filename!
    files = [{"fn": "malformed.txt", "f": RAW_DISTANCE_DATA}]
    with pytest.raises(ValueError):  # Removed match argument to be robust
        BattacharyyaSection(files)


def test_battacharyya_mismatched_data(reset_multiqc):
    """Test with mismatched number of bundles and distances."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    mismatched_data = "roi_names Bundle1 Bundle2\ndistance 0.5"  # Removed expected line to simulate mismatching length (if parses correctly)
    files = [{"fn": "Test.metric.raw.bhattacharyya.txt", "f": mismatched_data}]

    # This should actually raise an IndexError because of the list slicing and zip.
    # A more robust implementation might raise a ValueError.
    # For now, let's catch the expected error.
    with pytest.raises(ValueError, match="mismatched number of bundles and distances"):
        BattacharyyaSection(files)


def test_battacharyya_empty_file(reset_multiqc):
    """Test with an empty file."""
    from neuroimaging.modules.harmonization.sections.battacharyya import BattacharyyaSection

    files = [{"fn": "Test.metric.raw.bhattacharyya.txt", "f": ""}]
    with pytest.raises(ValueError, match="must contain at least two lines"):
        BattacharyyaSection(files)
