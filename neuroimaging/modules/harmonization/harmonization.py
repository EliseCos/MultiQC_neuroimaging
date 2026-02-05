import logging

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

from .sections.battacharyya import BattacharyyaSection
from .sections.distribution import DistributionSection
from .sections.selection import SelectionSection

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

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Load the reference stats file (useful to plot the reference scatterplot)
        ref_stats_file            = list(self.find_log_files(self.REF_STATS_SP_KEY))
        raw_stats_files           = list(self.find_log_files(self.RAW_STATS_SP_KEY))
        harmonized_stats_files    = list(self.find_log_files(self.HARMONIZED_STATS_SP_KEY))
        data_model_plots_files    = list(self.find_log_files(self.DATA_MODELS_PLOTS_SP_KEY))
        harmonization_plots_files = list(self.find_log_files(self.HARMONIZATION_PLOTS_SP_KEY))
        harmonization_distance_files = list(self.find_log_files(self.HARMONIZATION_DISTANCE_SP_KEY))

        dist_section_enabled = len(ref_stats_file) > 0 and \
                               len(raw_stats_files) > 0 and \
                               len(harmonized_stats_files) > 0 and \
                               len(data_model_plots_files) > 0 and \
                               len(harmonization_plots_files) > 0
        batt_section_enabled = len(harmonization_distance_files) > 0

        if not dist_section_enabled and not batt_section_enabled:
            log.debug("Missing required files for harmonization module for the distribution section and Bhattacharyya section."
                      " Skipping harmonization module.")
            raise ModuleNoSamplesFound

        select_section = SelectionSection()

        if dist_section_enabled:
            dist_section = DistributionSection(
                ref_stats_file,
                raw_stats_files,
                harmonized_stats_files,
                data_model_plots_files,
                harmonization_plots_files
            )
        if batt_section_enabled:
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
            if dist_section_enabled:
                dist_section.filter_bundles(bundles_filter)
            if batt_section_enabled:
                batt_section.filter_bundles(bundles_filter)
        if len(metrics_filter) > 0:
            if dist_section_enabled:
                dist_section.filter_metrics(metrics_filter)
            if batt_section_enabled:
                batt_section.filter_metrics(metrics_filter)

        # Check if there's any bundle/metric left after filtering
        if dist_section_enabled and batt_section_enabled:
            # Make sure that both sections have the same bundles and metrics
            assert set(dist_section.bundles) == set(batt_section.bundles), \
                "The bundles available in the distribution section and Bhattacharyya section do not match (after filtering)."
            assert set(dist_section.metrics) == set(batt_section.metrics), \
                "The metrics available in the distribution section and Bhattacharyya section do not match (after filtering)."
        
        # Empty list to prevent the SelectionSection from displaying the bundle dropdown if the distribution section is disabled
        # since the Bhattacharyya section only operates on the metrics.
        bundles = dist_section.bundles if dist_section_enabled else []
        metrics = dist_section.metrics if dist_section_enabled else batt_section.metrics

        # Make sure at least one bundle and metric is available after filtering
        if len(bundles) == 0 and len(metrics) == 0:
            log.debug("No bundle or metric available for plotting after filtering. Skipping harmonization module.")
            raise ModuleNoSamplesFound
        elif len(metrics) == 0:
            log.debug("No metric available for plotting after filtering. Skipping harmonization module.")
            raise ModuleNoSamplesFound
        elif dist_section_enabled and len(bundles) == 0:
            log.debug("No bundle available for plotting after filtering. Skipping the harmonization distribution results section.")
            dist_section_enabled = False
        
        select_section.set_bundles_metrics(bundles, metrics)

        if dist_section_enabled:
            dist_html = dist_section.build_html(select_section.default_bundle, select_section.default_metric, select_section.render_plot_func)
            select_section.register_hook_on_bundle_metric_change(dist_html.metadata["render_bundle_metric_hook"])

        if batt_section_enabled:
            batt_html = batt_section.build_html(select_section.default_metric, select_section.render_plot_func)
            select_section.register_hook_on_metric_change(batt_html.metadata["render_metric_hook"])
        
        # Build the selection section last to make sure we register the hooks and get the right metrics
        select_html = select_section.build_html()

        # Section with the dropdown menus to select the bundle and metric to display in the plots.
        self.add_section(
            name=select_section.name,
            anchor=select_section.anchor,
            description=select_section.description,
            content=select_html.content)

        # Section with the distribution plots (e.g. scatterplot of reference vs harmonized data, age curves, etc.)
        if dist_section_enabled:
            self.add_section(
                name=dist_section.name,
                anchor=dist_section.anchor,
                description=dist_section.description,
                content=dist_html.content)

        # Section with the Bhattacharyya distance boxplots
        if batt_section_enabled:
            self.add_section(
                name=batt_section.name,
                anchor=batt_section.anchor,
                description=batt_section.description,
                content=batt_html.content)
