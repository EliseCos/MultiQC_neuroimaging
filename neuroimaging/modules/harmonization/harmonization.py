import logging

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

from .sections.battacharyya import BattacharyyaSection
from .sections.distribution import DistributionSection

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")

#
# Configurable parameters:
#
# harmonization:
#     metrics: List of metrics to include in the harmonization or single metric string or "all" to include all metrics.
#     bundles: List of bundles to include in the harmonization or single bundle string or "all" to include all bundles.
#
# Search patterns required:
# sp:
#   harmonization/reference_stats:
#       fn: "*.reference.tsv"
#   harmonization/harmonized_stats:
#       fn: "*.harmonized.tsv"
#   harmonization/raw_stats:
#       - fn: "*mean_desc-roi_stats.tsv"
#         shared: true
#       - fn: "*bundles_mean_stats.tsv"
#         shared: true
#   harmonization/data_models_plots:
#       fn: "DataModels*.json"
#   harmonization/harmonization_plots:
#       fn: "AgeCurve*.json"
#   harmonization/harmonization_distance:
#       fn: "*.bhattacharrya.txt"

class MultiqcModule(BaseMultiqcModule):
    """
    This section contains several graphs to visualize the results of the harmonization
    procedure (e.g. using Clinical-Combat). The user can select the bundle and metric
    of interest using the dropdown menus at the top of the section, changing the plots
    displayed below accordingly.  
    Note: since this section requires features unsupported by the MultiQC native plotting
    library (e.g. filled areas between curves), we use Plotly to generate the plots. Thus,
    the sample highlighting features of MultiQC won't be available.
    """
    REF_STATS_SP_KEY            = "harmonization/reference_stats"
    RAW_STATS_SP_KEY            = "harmonization/raw_stats"
    HARMONIZED_STATS_SP_KEY     = "harmonization/harmonized_stats"
    DATA_MODELS_PLOTS_SP_KEY    = "harmonization/data_models_plots"
    HARMONIZATION_PLOTS_SP_KEY  = "harmonization/harmonization_plots"
    HARMONIZATION_DISTANCE_SP_KEY  = "harmonization/harmonization_distance"

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Harmonization results",
            anchor="harmonization",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=MultiqcModule.__doc__,
        )
        
        module_config = getattr(config, "harmonization", {})

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Load the reference stats file (useful to plot the reference scatterplot)
        ref_stats_file            = list(self.find_log_files(self.REF_STATS_SP_KEY))
        raw_stats_files           = list(self.find_log_files(self.RAW_STATS_SP_KEY))
        harmonized_stats_files    = list(self.find_log_files(self.HARMONIZED_STATS_SP_KEY))
        data_model_plots_files    = list(self.find_log_files(self.DATA_MODELS_PLOTS_SP_KEY))
        harmonization_plots_files = list(self.find_log_files(self.HARMONIZATION_PLOTS_SP_KEY))
        harmonization_distance_files = list(self.find_log_files(self.HARMONIZATION_DISTANCE_SP_KEY))

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(harmonized_stats_files) == 0:
            log.debug(f"Could not find harmonization reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        dist_section = DistributionSection(
            ref_stats_file,
            raw_stats_files,
            harmonized_stats_files,
            data_model_plots_files,
            harmonization_plots_files
        )

        batt_section = BattacharyyaSection(
            harmonization_distance_files
        )

        # Fetch the bundles to render from the configuration
        bundles_filter = module_config.get("bundles", [])
        if isinstance(bundles_filter, str) and bundles_filter != "all":
            bundles_filter = [bundles_filter]

        # Fetch the bundles to render from the configuration
        metrics_filter = module_config.get("metrics", ["fa", "md", "rd", "ad"])
        if isinstance(metrics_filter, str) and metrics_filter != "all":
            metrics_filter = [metrics_filter]

        if len(bundles_filter) > 0:
            dist_section.filter_bundles(bundles_filter)
            batt_section.filter_bundles(bundles_filter)
        if len(metrics_filter) > 0:
            dist_section.filter_metrics(metrics_filter)
            batt_section.filter_metrics(metrics_filter)

        # Make sure that both sections have the same bundles and metrics
        assert set(dist_section.bundles) == set(batt_section.bundles), \
            "The bundles available in the distribution section and Bhattacharyya section do not match (after filtering)."
        assert set(dist_section.metrics) == set(batt_section.metrics), \
            "The metrics available in the distribution section and Bhattacharyya section do not match (after filtering)."

        dist_html = dist_section.build_html()

        self.add_section(
            name=dist_section.name,
            anchor=dist_section.anchor,
            description=dist_section.description,
            content=dist_html.content)

        batt_html = batt_section.build_html(
            dist_html.metadata["default_metric"],
            dist_html.metadata["render_bhatt_func"],
            dist_html.metadata["render_plot_func"])

        self.add_section(
            name=batt_section.name,
            anchor=batt_section.anchor,
            description=batt_section.description,
            content=batt_html.content)

