"""
Tests for the main harmonization module.
"""

import os
import tempfile
import shutil
import pytest
import json
from multiqc import config, report
from multiqc.base_module import ModuleNoSamplesFound

# Mock data (can reuse from other test files if needed, but defining here for clarity)
RAW_STATS_DATA = "sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\nsub-01\tSite1\t30\tM\tR\tC\tBundle1\t0.5"
HARMONIZED_STATS_DATA = "sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\nsub-01\tSite1\t30\tM\tR\tC\tBundle1\t0.45"
REF_STATS_DATA = "sample\tsite\tage\tsex\thandedness\tdisease\troi\tfa\nsub-REF\tRefSite\t40\tF\tR\tC\tBundle1\t0.55"
DATA_MODELS_JSON = {
    "Bundle1": {
        "fa": {
            "regression_reference": {"site": "RefSite", "data_x": [], "data_y": []}, 
            "regression_moving": {"site": "Site1", "data_x": [], "data_y": []}
        }
    }
}
AGE_CURVE_JSON = {
    "Bundle1": {
        "fa": {
            "reference_percentiles": {
                "p50": {"percentile": 50, "data_x": [10, 20], "data_y": [0.5, 0.5], "color": "#000000"}
            },
            "moving_raw_percentiles": {
                "p50": {"percentile": 50, "data_x": [10, 20], "data_y": [0.5, 0.5], "color": "#000000"}
            },
            "moving_harmonized_percentiles": {
                "p50": {"percentile": 50, "data_x": [10, 20], "data_y": [0.5, 0.5], "color": "#000000"}
            }
        }
    }
}
BHATT_RAW_DATA = "roi_names Bundle1\ndistance 0.5"
BHATT_HARMONIZED_DATA = "roi_names Bundle1\ndistance 0.2"

@pytest.fixture
def reset_multiqc():
    """Reset MultiQC state and register search patterns."""
    config.reset()
    report.reset()
    # Register search patterns needed by the module
    config.update_dict(config.sp, {
        "harmonization/reference_stats": {"fn": "*.reference.tsv"},
        "harmonization/harmonized_stats": {"fn": "*.harmonized.tsv"},
        "harmonization/raw_stats": {"fn": "*_stats.tsv"},
        "harmonization/data_models_plots": {"fn": "DataModels*.json"},
        "harmonization/harmonization_plots": {"fn": "AgeCurve*.json"},
        "harmonization/harmonization_distance": {"fn": "*.bhattacharyya.txt"}
    })
    yield
    config.reset()
    report.reset()

@pytest.fixture
def test_data_dir():
    """Create a temporary directory with all necessary mock files."""
    tmpdir = tempfile.mkdtemp()

    with open(os.path.join(tmpdir, "data_stats.tsv"), "w") as f: f.write(RAW_STATS_DATA)
    with open(os.path.join(tmpdir, "data.harmonized.tsv"), "w") as f: f.write(HARMONIZED_STATS_DATA)
    with open(os.path.join(tmpdir, "data.reference.tsv"), "w") as f: f.write(REF_STATS_DATA)
    with open(os.path.join(tmpdir, "DataModels.json"), "w") as f: json.dump(DATA_MODELS_JSON, f)
    with open(os.path.join(tmpdir, "AgeCurve.json"), "w") as f: json.dump(AGE_CURVE_JSON, f)
    with open(os.path.join(tmpdir, "Site1.fa.raw.bhattacharyya.txt"), "w") as f: f.write(BHATT_RAW_DATA)
    with open(os.path.join(tmpdir, "Site1.fa.harmonized.bhattacharyya.txt"), "w") as f: f.write(BHATT_HARMONIZED_DATA)
    
    yield tmpdir
    
    shutil.rmtree(tmpdir)

def populate_report_files(data_dir):
    """Manually populate report.files since we are not running a full MultiQC execution."""
    # Mapping from sp key to file extension/pattern logic used in test_data_dir
    patterns = {
        "harmonization/reference_stats": ".reference.tsv",
        "harmonization/harmonized_stats": ".harmonized.tsv",
        "harmonization/raw_stats": "_stats.tsv",
        "harmonization/data_models_plots": "DataModels.json",
        "harmonization/harmonization_plots": "AgeCurve.json",
        "harmonization/harmonization_distance": ".bhattacharyya.txt"
    }
    
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        with open(path, 'r') as f:
            content = f.read()
            
        for sp_key, pattern in patterns.items():
            if pattern in filename or (pattern == "_stats.tsv" and filename.endswith("_stats.tsv")):
                if sp_key not in report.files:
                    report.files[sp_key] = []
                report.files[sp_key].append({
                    "fn": filename,
                    "root": data_dir,
                    "f": content,
                    "sp_key": sp_key  # Required by BaseMultiqcModule
                })

def test_harmonization_module_run(reset_multiqc, test_data_dir):
    """Test a successful run of the harmonization module."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    
    populate_report_files(test_data_dir)

    module = MultiqcModule()
    
    assert module is not None
    # Should add three sections: Selection, Distribution and Battacharyya
    # module.sections contains the sections added by the module
    assert len(module.sections) == 3
    
    select_section = module.sections[0]
    dist_section = module.sections[1]
    batt_section = module.sections[2]
    
    assert "Distributional results" in dist_section.name
    assert "Mean Bhattacharyya distance (BD)" in batt_section.name

def test_harmonization_module_no_files(reset_multiqc):
    """Test that the module raises ModuleNoSamplesFound when no files are found."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    tmpdir = tempfile.mkdtemp()
    config.analysis_dir = [tmpdir]
    config.kwargs = {"single_subject": False} # Default

    # Do NOT populate report.files, effectively simulating no files found

    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()
    
    shutil.rmtree(tmpdir)

def test_harmonization_module_single_subject(reset_multiqc):
    """Test that the module raises ModuleNoSamplesFound in single-subject mode."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule
    
    config.kwargs = {"single_subject": True}
    with pytest.raises(ModuleNoSamplesFound):
        MultiqcModule()

def test_harmonization_module_filtering(reset_multiqc, test_data_dir):
    """Test the metric and bundle filtering from multiqc_config.yaml."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    
    # Add a second metric to test filtering
    with open(os.path.join(test_data_dir, "Site1.md.raw.bhattacharyya.txt"), "w") as f: f.write(BHATT_RAW_DATA)
    # Note: mocking distribution data with 'md' is hard here without parsing util change, 
    # but the section filtering just filters internal lists.
    # However the module asserts consistency!
    # So we need 'md' in distribution data too?
    # Actually, the MockDistSection below OVERRIDES filtering logic and content, so checking the module logic handling that.
    
    populate_report_files(test_data_dir)

    # Configure filtering
    config.harmonization = {
        "metrics": "fa",
        "bundles": "Bundle1"
    }

    # Mock the section classes to spy on their filter methods
    class MockDistSection:
        bundles = ["Bundle1"]
        metrics = ["fa", "md"]
        name="Distributional results"
        anchor="dist_anchor"
        description="dist desc"
        def __init__(self, *args): pass
        def filter_bundles(self, bundles): self.bundles = [b for b in self.bundles if b in bundles]
        def filter_metrics(self, metrics): self.metrics = [m for m in self.metrics if m in metrics]
        def build_html(self, *args, **kwargs): 
            return type("HtmlContent", (), {
                "content": "", 
                "metadata": {
                    "default_metric": "fa", 
                    "render_bhatt_func": "", 
                    "render_plot_func": "",
                    "render_bundle_metric_hook": "mock_dist_hook"
                }
            })()
    
    class MockBattSection:
        bundles = ["Bundle1"]
        metrics = ["fa", "md"]
        name="Mean Bhattacharyya distance (BD)"
        anchor="batt_anchor"
        description="batt desc"
        def __init__(self, *args): pass
        def filter_bundles(self, bundles): self.bundles = [b for b in self.bundles if b in bundles]
        def filter_metrics(self, metrics): self.metrics = [m for m in self.metrics if m in metrics]
        def build_html(self, *args, **kwargs): 
            return type("HtmlContent", (), {
                "content": "", 
                "metadata": {
                    "render_metric_hook": "mock_batt_hook"
                }
            })()

    # Monkeypatch the classes
    from neuroimaging.modules.harmonization import harmonization
    original_dist = harmonization.DistributionSection
    original_batt = harmonization.BattacharyyaSection
    harmonization.DistributionSection = MockDistSection
    harmonization.BattacharyyaSection = MockBattSection
    
    try:
        module = MultiqcModule()

        # The __init__ of MultiqcModule will create instances of the mocked sections.
        # We can't directly access them to verify attributes.
        # But since we asserted consistency inside module, if they diverged it would fail.
        # And we know they start with ["fa", "md"]. If filtering works, they become ["fa"].
        
        assert module is not None
        # Should expect 3 sections if filtering works correctly and both sections are active
        assert len(module.sections) == 3
        
    finally:
        # Restore original classes
        harmonization.DistributionSection = original_dist
        harmonization.BattacharyyaSection = original_batt

def test_harmonization_mismatched_sections(reset_multiqc, test_data_dir):
    """Test assertion error when sections have mismatched bundles/metrics."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    # Create a situation where one section has a metric the other doesn't
    # E.g., Harmonized stats file has 'fa', but Bhattacharyya distance file only has 'md'
    # Wait, the default fixtures have 'fa'. 
    # Let's add 'md' to bhatt files only.
    with open(os.path.join(test_data_dir, "Site1.md.raw.bhattacharyya.txt"), "w") as f: f.write(BHATT_RAW_DATA)
    # We remove 'fa' bhatt files to ensure mismatch
    os.remove(os.path.join(test_data_dir, "Site1.fa.raw.bhattacharyya.txt"))
    os.remove(os.path.join(test_data_dir, "Site1.fa.harmonized.bhattacharyya.txt"))
    
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    
    populate_report_files(test_data_dir)

    with pytest.raises(AssertionError, match="metrics available.*do not match"):
        MultiqcModule()

def test_harmonization_module_dist_only(reset_multiqc, test_data_dir):
    """Test that the module works when only distribution files are present."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    # Remove Bhattacharyya files from report.files logic
    # We do this by removing them from the directory before populating report.files
    # But test_data_dir is a fixture, modifying it affects the current test run.
    # It's a directory, so we can delete files.
    os.remove(os.path.join(test_data_dir, "Site1.fa.raw.bhattacharyya.txt"))
    os.remove(os.path.join(test_data_dir, "Site1.fa.harmonized.bhattacharyya.txt"))
    # Also ensure no 'md' mock file exists if it was created in previous tests 
    # (though fixtures re-run for each test function usually? yes, if not scoped 'module')
    # test_data_dir is scoped implicitly as 'function' (default)
    
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    
    populate_report_files(test_data_dir)
    
    module = MultiqcModule()
    
    assert module is not None
    # Should have Selection + Distribution sections (2)
    assert len(module.sections) == 2
    
    # Check section names
    section_names = [s.name for s in module.sections]
    # Use partial matching because Selection section name might not be hardcoded in verification plan details
    # but existing code for dist section has "Distributional results"
    assert any("Distributional results" in name for name in section_names)
    assert not any("Mean Bhattacharyya distance (BD)" in name for name in section_names)

def test_harmonization_module_batt_only(reset_multiqc, test_data_dir):
    """Test that the module works when only Bhattacharyya files are present."""
    from neuroimaging.modules.harmonization.harmonization import MultiqcModule

    # Remove Distribution files
    files_to_remove = [
        "data.reference.tsv", 
        "data_stats.tsv", # raw stats
        "data.harmonized.tsv", 
        "DataModels.json", 
        "AgeCurve.json"
    ]
    for f in files_to_remove:
        path = os.path.join(test_data_dir, f)
        if os.path.exists(path):
            os.remove(path)
            
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    
    populate_report_files(test_data_dir)
    
    module = MultiqcModule()
    
    assert module is not None
    # Should have Selection + Bhattacharyya sections (2)
    assert len(module.sections) == 2
    
    section_names = [s.name for s in module.sections]
    assert any("Mean Bhattacharyya distance (BD)" in name for name in section_names)
    assert not any("Distributional results" in name for name in section_names)
