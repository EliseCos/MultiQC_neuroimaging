"""
Tests for the harmonization module Distribution section.
"""

import os
import tempfile
import shutil
import pytest
import json
from multiqc import config, report

# Mock data for stats files
REF_STATS_DATA = """\
sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\tmd
sub-REF01\tRefSite\t30\tM\tR\tControl\tBundle1\t0.5\t0.001
"""

RAW_STATS_DATA = """\
sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\tmd
sub-MOV01\tMovSite\t32\tF\tR\tControl\tBundle1\t0.6\t0.002
sub-MOV02\tMovSite\t35\tM\tL\tControl\tBundle1\t0.65\t0.0022
"""

HARMONIZED_STATS_DATA = """\
sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\tmd
sub-MOV01\tMovSite\t32\tF\tR\tControl\tBundle1\t0.51\t0.0011
sub-MOV02\tMovSite\t35\tM\tL\tControl\tBundle1\t0.54\t0.0013
"""

# Mock data for plot JSONs
DATA_MODELS_JSON = {
    "Bundle1": {
        "fa": {
            "regression_reference": {"site": "RefSite", "data_x": [20, 80], "data_y": [0.4, 0.6], "color": [255, 0, 0]},
            "regression_moving": {"site": "MovSite", "data_x": [20, 80], "data_y": [0.5, 0.7], "color": [0, 0, 255]},
        },
        "md": {
            "regression_reference": {
                "site": "RefSite",
                "data_x": [20, 80],
                "data_y": [0.001, 0.002],
                "color": [255, 0, 0],
            },
            "regression_moving": {
                "site": "MovSite",
                "data_x": [20, 80],
                "data_y": [0.002, 0.003],
                "color": [0, 0, 255],
            },
        },
    }
}

AGE_CURVE_JSON = {
    "Bundle1": {
        "fa": {
            "reference_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.3, 0.5], "color": "#FF0000"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.4, 0.6], "color": "#FF0000"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.5, 0.7], "color": "#FF0000"},
            },
            "moving_raw_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.4, 0.6], "color": "#0000FF"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.5, 0.7], "color": "#0000FF"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.6, 0.8], "color": "#0000FF"},
            },
            "moving_harmonized_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.32, 0.52], "color": "#00FF00"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.42, 0.62], "color": "#00FF00"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.52, 0.72], "color": "#00FF00"},
            },
        },
        "md": {
            "reference_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.001, 0.003], "color": "#FF0000"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.002, 0.004], "color": "#FF0000"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.003, 0.005], "color": "#FF0000"},
            },
            "moving_raw_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.001, 0.003], "color": "#0000FF"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.002, 0.004], "color": "#0000FF"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.003, 0.005], "color": "#0000FF"},
            },
            "moving_harmonized_percentiles": {
                "p5": {"percentile": 5, "data_x": [20, 80], "data_y": [0.001, 0.003], "color": "#00FF00"},
                "p50": {"percentile": 50, "data_x": [20, 80], "data_y": [0.002, 0.004], "color": "#00FF00"},
                "p95": {"percentile": 95, "data_x": [20, 80], "data_y": [0.003, 0.005], "color": "#00FF00"},
            },
        },
    }
}


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

    # Write stats files
    with open(os.path.join(tmpdir, "ref.reference.tsv"), "w") as f:
        f.write(REF_STATS_DATA)
    with open(os.path.join(tmpdir, "raw.mean_desc-roi_stats.tsv"), "w") as f:
        f.write(RAW_STATS_DATA)
    with open(os.path.join(tmpdir, "harm.harmonized.tsv"), "w") as f:
        f.write(HARMONIZED_STATS_DATA)

    # Write JSON files
    with open(os.path.join(tmpdir, "DataModels.json"), "w") as f:
        json.dump(DATA_MODELS_JSON, f)
    with open(os.path.join(tmpdir, "AgeCurve.json"), "w") as f:
        json.dump(AGE_CURVE_JSON, f)

    yield tmpdir

    shutil.rmtree(tmpdir)


def get_dist_files(test_data_dir):
    """Helper to create file lists for the section."""
    ref_stats_file = []
    raw_stats_files = []
    harmonized_stats_files = []
    data_model_plots_files = []
    harmonization_plots_files = []

    for fn in os.listdir(test_data_dir):
        path = os.path.join(test_data_dir, fn)
        with open(path, "r") as f:
            content = f.read()
            if "reference.tsv" in fn:
                ref_stats_file.append({"fn": fn, "f": content})
            elif "roi_stats.tsv" in fn:
                raw_stats_files.append({"fn": fn, "f": content})
            elif "harmonized.tsv" in fn:
                harmonized_stats_files.append({"fn": fn, "f": content})
            elif "DataModels.json" in fn:
                data_model_plots_files.append({"fn": fn, "f": content})
            elif "AgeCurve.json" in fn:
                harmonization_plots_files.append({"fn": fn, "f": content})

    return ref_stats_file, raw_stats_files, harmonized_stats_files, data_model_plots_files, harmonization_plots_files


def test_distribution_section_init(reset_multiqc, test_data_dir):
    """Test the DistributionSection constructor with valid data."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection

    files = get_dist_files(test_data_dir)
    section = DistributionSection(*files)

    assert "fa" in section.metrics
    assert "md" in section.metrics
    assert "Bundle1" in section.bundles

    assert not section.ref_df.empty
    assert not section.raw_df.empty
    assert not section.harmonized_df.empty
    assert "Bundle1" in section.data_model_json.data
    assert "Bundle1" in section.harmonization_json.data


def test_distribution_section_mismatched_samples(reset_multiqc, test_data_dir):
    """Test constructor with mismatched samples between raw and harmonized data."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection

    # Modify harmonized data to have different samples
    bad_harmonized_data = HARMONIZED_STATS_DATA.replace("sub-MOV01", "sub-DIFFERENT")
    with open(os.path.join(test_data_dir, "harm.harmonized.tsv"), "w") as f:
        f.write(bad_harmonized_data)

    files = get_dist_files(test_data_dir)
    with pytest.raises(AssertionError, match="must contain the same samples"):
        DistributionSection(*files)


def test_distribution_section_mismatched_rois(reset_multiqc, test_data_dir):
    """Test constructor with mismatched ROIs."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection

    # Modify harmonized data to have different ROI
    bad_harmonized_data = HARMONIZED_STATS_DATA.replace("Bundle1", "Bundle2")
    with open(os.path.join(test_data_dir, "harm.harmonized.tsv"), "w") as f:
        f.write(bad_harmonized_data)

    files = get_dist_files(test_data_dir)
    with pytest.raises(AssertionError, match="must contain the same ROIs"):
        DistributionSection(*files)


def test_distribution_filtering(reset_multiqc, test_data_dir):
    """Test filtering of metrics and bundles."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection

    files = get_dist_files(test_data_dir)
    section = DistributionSection(*files)

    # Add another metric/bundle for robust testing
    section.metrics.add("extra_metric")
    section.bundles.append("extra_bundle")

    # Filter metrics
    section.filter_metrics(["fa"])
    assert section.metrics == {"fa"}

    # Filter bundles
    section.filter_bundles(["Bundle1"])
    assert section.bundles == ["Bundle1"]


def test_distribution_percentile_plotter(reset_multiqc):
    """Test the _plot_percentiles_and_get_median_curve helper function."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    section = object.__new__(DistributionSection)  # Create instance without calling __init__

    # Test with even number of percentiles
    even_percentiles = {"p10": {"percentile": 10}, "p90": {"percentile": 90}}
    with pytest.raises(ValueError, match="number of percentiles must be odd"):
        section._plot_percentiles_and_get_median_curve(even_percentiles, None, 0, 0, False)

    # Test with valid odd number of percentiles
    odd_percentiles = {
        "p5": {"percentile": 5, "data_x": [1], "data_y": [1], "color": "red"},
        "p50": {"percentile": 50},
        "p95": {"percentile": 95, "data_x": [1], "data_y": [1], "color": "red"},
    }

    mock_fig = make_subplots(rows=1, cols=1)
    median_key = section._plot_percentiles_and_get_median_curve(odd_percentiles, mock_fig, 1, 1, False)

    assert median_key == "p50"
    assert len(mock_fig.data) == 1  # Should have added one trace for the fill
    assert mock_fig.data[0].fill == "toself"


def test_distribution_build_html(reset_multiqc, test_data_dir):
    """Test the build_html method."""
    from neuroimaging.modules.harmonization.sections.distribution import DistributionSection

    files = get_dist_files(test_data_dir)
    section = DistributionSection(*files)

    html_content = section.build_html(
        default_bundle=section.bundles[0], default_metric=list(section.metrics)[0], render_plot_func="renderPlot"
    )

    assert html_content is not None
    assert isinstance(html_content.content, str)

    # Check that plotly graph divs are present
    assert "plotly-graph-div" in html_content.content

    # Check metadata
    assert "render_bundle_metric_hook" in html_content.metadata
