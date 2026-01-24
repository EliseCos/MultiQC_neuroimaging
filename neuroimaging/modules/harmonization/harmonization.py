"""MultiQC module for harmonization results."""

import os
import csv
import logging
from typing import Dict, Any
from collections import defaultdict

import numpy as np

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import linegraph, bargraph, scatter
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import json

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")

pconfig = {
    # Building the plot
    "id": "scatter_1",  # HTML ID used for plot
    # "categories": False,         # Set to True to use x values as categories instead of numbers.
    # "colors": dict(),            # Provide dict with keys = sample names and values colours
    # "smooth_points": None,       # Supply a number to limit number of points / smooth data
    # "smooth_points_sumcounts": True,  # Sum counts in bins, or average? Can supply list for multiple datasets
    # "logswitch": False,          # Show the 'Log10' switch?
    # "logswitch_active": False,   # Initial display with 'Log10' active?
    # "logswitch_label": "Log10",  # Label for 'Log10' button
    # "axis_controlled_by_switches": ["yaxis"], # Which axes should be impacted by the switch button (one or both of xaxis, yaxis)
    # "extra_series": None,        # See section below
    # # Plot configuration
    "title": "Harmonization: Clinical combat distributional results",  # Plot title - should be in format "Module Name: Plot Title"
    # "xlab": None,                # X axis label
    # "ylab": None,                # Y axis label
    # "xmax": None,                # Hard max x limit
    # "xmin": None,                # Hard min x limit
    # "ymax": None,                # Hard max y limit
    # "ymin": None,                # Hard min y limit
    # "x_clipmax": None,           # Max value allowed for automatic axis limit
    # "x_clipmin": None,           # Min value allowed for automatic axis limit
    # "y_clipmax": None,           # Max value allowed for automatic axis limit
    # "y_clipmin": None,           # Min value allowed for automatic axis limit
    # "x_minrange": None,          # Min range for x-axis (5 would allow 0..5, but also 15..20, etc.)
    # "y_minrange": None,          # Min range for y-axis (5 would allow 0..5, but also 15..20, etc.)
    # "xlog": False,               # Use log10 for the x-axis
    # "ylog": False,               # Use log10 scale for the y-axis
    # "y_bands": None,             # Horizontal colored background bands
    # "x_bands": None,             # Vertical colored background bands
    # "y_lines": None,             # Extra horizontal lines
    # "x_lines": None,             # Extra vertical lines
    # "xsuffix": "%",              # Suffix for the X-axis values and labels. Parsed from tt_label by default
    # "ysuffix": "%",              # Suffix for the Y-axis values and labels. Parsed from tt_label by default
    # "tt_label": "{x}: {y:.2f}",  # Customise tooltip label, e.g. '{point.x} base pairs'
    # "tt_decimals": None,         # Tooltip decimals when categories = True (when false use tt_label)
    # "height": 500,               # The default height of the plot, in pixels
    # "style": "line",             # The style of the line. Can be "line" or "lines+markers"
    # "square": False,             # Force the plot to stay square? (Maintain aspect ratio)
}


def gen_line_data(base, variance):
    return {
        "Sample_A": {i: base + (i * 2) + random.uniform(-variance, variance) for i in range(1, 6)},
        "Sample_B": {i: base + (i * 1.5) + random.uniform(-variance, variance) for i in range(1, 6)},
        "Sample_C": {i: base - (i * 0.5) + random.uniform(-variance, variance) for i in range(1, 6)},
    }


def gen_scatter_data(base, variance):
    return {
        "Sample_A": {"x": 2 + random.uniform(-variance, variance), "y": 20 + random.uniform(-variance, variance)},
        "Sample_B": {"x": 3 + random.uniform(-variance, variance), "y": 25 + random.uniform(-variance, variance)},
        "Sample_C": {"x": 4 + random.uniform(-variance, variance), "y": 15 + random.uniform(-variance, variance)},
    }


class MultiqcModule(BaseMultiqcModule):
    """
    This section aims to visualize the result of the harmonization process.
    """

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Harmonization distribution results",
            anchor="harmonization",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=self.__doc__,
        )

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Find files using the custom search pattern added in custom_code
        files = list(self.find_log_files("harmonization"))

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(files) == 0:
            log.debug(f"Could not find harmonization reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        meta, data = self._parse_tsv_file(files[0])

        # --- 1. Prepare Your Data ---
        # Assume you have a dictionary of all your tuples
        # Format: { "Tuple_Name": [Data_For_Plot_1, Data_For_Plot_2, Data_For_Plot_3] }
        metrics_bundles = {
            "Bundle 1": {
                "Metric 1": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 2": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 3": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
            },
            "Bundle 2": {
                "Metric 1": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 2": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 3": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
            },
            "Bundle 3": {
                "Metric 1": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 2": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
                "Metric 3": [
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                    gen_scatter_data(base=15, variance=0.5),
                ],
            },
        }

        # --- 2. Render the INITIAL View (The First Tuple) ---
        # We use fixed IDs (p1_container, p2_container...) so we can find them with JS later

        def add_regression(plot_data):
            x_vals = np.array([d["x"] for d in plot_data.values()])
            y_vals = np.array([d["y"] for d in plot_data.values()])

            # Linear fit
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            y_fit = coeffs[0] * x_range + coeffs[1]

            # Calculate standard error for confidence bands
            y_pred = coeffs[0] * x_vals + coeffs[1]
            residuals = y_vals - y_pred
            std_error = np.sqrt(np.sum(residuals**2) / (len(x_vals) - 2))
            y_upper = y_fit + 2 * std_error
            y_lower = y_fit - 2 * std_error

            return {
                "extra_series": [
                    {
                        "name": f"Linear fit (R² = {np.corrcoef(x_vals, y_vals)[0, 1] ** 2:.3f})",
                        "data": [[float(x), float(y)] for x, y in zip(x_range, y_fit)],
                        "color": "#ff0000",
                        "dashStyle": "Dash",
                        "zoneAxis": "x",
                        "bands": [
                            [
                                [float(x), float(y_low), float(y_high)]
                                for x, y_low, y_high in zip(x_range, y_lower, y_upper)
                            ]
                        ],
                    }
                ]
            }

        # Create the Flexbox Layout for these 3 plots
        def make_flex_row(plot_data_list, bundle, metric, hidden=False):
            div_id = f"{bundle.replace(' ', '-')}_{metric.replace(' ', '-')}"
            # Generate the 3 individual plots
            # Note: Use unique IDs to prevent DOM conflicts!
            plot_anchors = [f"p1_{div_id}", f"p2_{div_id}", f"p3_{div_id}"]
            p1 = scatter.plot(
                plot_data_list[0],
                pconfig={
                    "id": plot_anchors[0],
                    "title": f"Datamodel raw ({metric} in {bundle})",
                    **add_regression(plot_data_list[0]),
                },
            )
            p2 = scatter.plot(
                plot_data_list[1],
                pconfig={
                    "id": plot_anchors[1],
                    "title": f"AgeCurve raw ({metric} in {bundle})",
                    **add_regression(plot_data_list[1]),
                },
            )
            p3 = scatter.plot(
                plot_data_list[2],
                pconfig={
                    "id": plot_anchors[2],
                    "title": f"AgeCurve harmonized ({metric} in {bundle})",
                    **add_regression(plot_data_list[2]),
                },
            )

            p1_html = p1.add_to_report(module_anchor="harmonization", section_anchor="harmonization_distributions")
            p2_html = p2.add_to_report(module_anchor="harmonization", section_anchor="harmonization_distributions")
            p3_html = p3.add_to_report(module_anchor="harmonization", section_anchor="harmonization_distributions")

            return (
                f"""
            <div id="{div_id}" style="display: {"none" if hidden else "flex"}; flex-direction: row; justify-content: space-between;">
                <div style="width: 32%;">{p1_html}</div>
                <div style="width: 32%;">{p2_html}</div>
                <div style="width: 32%;">{p3_html}</div>
            </div>
            """,
                {"plot_anchors": plot_anchors, "div_id": div_id},
                div_id,
            )

        # --- 3. Generate content for each tab ---
        is_first_tuple = True
        all_div_ids = []
        anchors = {}
        plots_html_content = ""
        for bundle, metrics in metrics_bundles.items():
            for metric, data in metrics.items():
                html, plot_anchors, div_id = make_flex_row(data, bundle, metric, hidden=not is_first_tuple)

                plots_html_content += html

                # Keep the div IDs and anchors for JS switching
                all_div_ids.append(div_id)

                if bundle not in anchors:
                    anchors[bundle] = {}
                anchors[bundle][metric] = plot_anchors
                is_first_tuple = False

        html_bundle_options = "".join(
            [
                f'<option value="{bundle}" {"selected" if i == 0 else ""}>{bundle}</option>'
                for i, bundle in enumerate(metrics_bundles.keys())
            ]
        )
        html_metric_options = "".join(
            [
                f'<option value="{metric}" {"selected" if i == 0 else ""}>{metric}</option>'
                for i, metric in enumerate(next(iter(metrics_bundles.values())).keys())
            ]
        )
        default_bundle = list(metrics_bundles.keys())[0]
        default_metric = list(next(iter(metrics_bundles.values())).keys())[0]

        html_script = f"""
        <div class="d-flex justify-content-center gap-3">
            <select class="form-select w-auto" aria-label="Metric selection" onchange="showScenario(current_bundle, this.value)">
                {html_metric_options}
            </select>
            
            <select class="form-select w-auto" aria-label="Bundle selection" onchange="showScenario(this.value, current_metric)">
                {html_bundle_options}
            </select>
        </div>
        <script>
        var anchors = {json.dumps(anchors)};
        var div_ids = {json.dumps(all_div_ids)};
        var current_bundle = "{default_bundle}";
        var current_metric = "{default_metric}";

        function showScenario(bundle, metric) {{
            console.log("Switching to view: ", bundle, ",", metric);
            current_bundle = bundle;
            current_metric = metric;

            // Make sure all plot divs are hidden before showing the selected one
            div_ids.forEach(function(divId) {{
                document.getElementById(divId).style.display = 'none';
            }});
            
            // Show the selected scenario
            var selected_div_id = anchors[bundle][metric]['div_id'];
            console.log("Showing div ID:", selected_div_id);
            document.getElementById(selected_div_id).style.display = 'flex';
            
            var plotIds = anchors[bundle][metric]['plot_anchors'];
            console.log("Rendering plots with IDs:", plotIds);
            plotIds.forEach(function(plotId) {{
                // The renderPlot function is provided by MultiQC's default template.
                // It handles the rendering of plots by their IDs (i.e. plot anchors).
                // If the plots are not visible when the window is loaded, they simply
                // won't be rendered. When we switch the metrics/experiments, we need to
                // explicitly call renderPlot again to make sure that they get drawn.
                renderPlot(plotId);
            }}); 
        }}
        </script>
        """

        html_content = html_script + plots_html_content

        # --- 4. Add Section with Tabs ---
        # MultiQC automatically creates tabs if you pass a dict to 'plot'
        self.add_section(name="Comparisons", anchor="my_comparison_section", content=html_content)

    def _fetch_data_for_roi_metric(
        self, data: Dict[str, Dict[str, Dict[str, Any]]], roi: str, metric: str
    ) -> Dict[str, float]:
        """Fetch data for a specific ROI and metric across all samples."""
        plot_data = {}
        for sample, rois in data.items():
            if roi in rois and metric in rois[roi]:
                plot_data[sample] = rois[roi][metric]
        return plot_data

    def _parse_tsv_file(self, f) -> Dict[str, Any]:
        """Parse a TSV file and return its contents as a dictionary."""
        data = defaultdict(lambda: defaultdict(dict))
        meta = defaultdict(dict)
        # data = {
        #     "sub-01": {
        #         "roi_1": {
        #             "FA": 0.45,
        #             "MD": 0.0008
        #         },
        #         "roi_2": {
        #             "FA": 0.50,
        #             "MD": 0.0007
        #         }
        #     },
        #     "sub-02": {
        #         "roi_1": {
        #             "FA": 0.47,
        #             "MD": 0.0009
        #         },
        #         "roi_2": {
        #             "FA": 0.52,
        #             "MD": 0.0006
        #         }
        #     }
        # }

        file_content = f.get("f", "").splitlines()
        reader = csv.DictReader(file_content, delimiter="\t")
        headers = set(reader.fieldnames)

        meta_columns = set(["site", "age", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        metric_columns = headers - non_metric_columns
        for row in reader:
            sample = row["sample"]
            roi = row["roi"]

            if not meta.get(sample, None):
                # Use dict comprehension to populate meta information
                meta[sample] = {col: row[col] for col in meta_columns}

            for metric in metric_columns:
                value = float(row[metric])
                data[sample][roi][metric] = value

        return dict(meta), dict(data)

    def _parse_tsv_file_sections(self, f) -> Dict[str, Any]:
        """Parse a TSV file and return its contents as a dictionary."""
        data = defaultdict(lambda: defaultdict(dict))
        meta = defaultdict(dict)
        # data = {
        #     "bundle_1": {
        #         "roi_1": {
        #             "sample-01": 0.45,
        #             "sample-02": 0.0008
        #         },
        #         "roi_2": {
        #             "sample-02": 0.50,
        #             "sample-02": 0.0007
        #         }
        #     },
        #     "bundle_2": {
        #         "roi_1": {
        #             "FA": 0.47,
        #             "MD": 0.0009
        #         },
        #         "roi_2": {
        #             "FA": 0.52,
        #             "MD": 0.0006
        #         }
        #     }
        # }

        file_content = f.get("f", "").splitlines()
        reader = csv.DictReader(file_content, delimiter="\t")
        headers = set(reader.fieldnames)

        meta_columns = set(["site", "age", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        metric_columns = headers - non_metric_columns
        for row in reader:
            sample = row["sample"]
            roi = row["roi"]

            if not meta.get(sample, None):
                # Use dict comprehension to populate meta information
                meta[sample] = {col: row[col] for col in meta_columns}

            for metric in metric_columns:
                value = float(row[metric])
                data[sample][roi][metric] = value

        return dict(meta), dict(data)


class ClaudeMultiqcModule(BaseMultiqcModule):
    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Example Module", anchor="example_module", info="Example showing 3 plots with single shared dropdown"
        )

        # Generate fake data for 10 different groups
        x_values = list(range(1, 101))

        # Store all group data
        all_groups = []

        # Generate 10 groups, each with 3 plots
        for group_num in range(1, 11):
            group_plots = []
            for plot_num in range(1, 4):
                base = group_num * 5 + plot_num * 10
                noise = 2 + group_num * 0.3
                data = self._generate_sample_data(x_values, base=base, noise=noise)
                group_plots.append(data)
            all_groups.append(group_plots)

        # Create the three plots with synchronized switching
        self._create_plots(all_groups)

    def _generate_sample_data(self, x_values, base=10, noise=2):
        """Generate fake sample data with some trend and noise"""
        samples = {}
        for i in range(3):
            sample_name = f"Sample_{i + 1}"
            y_values = [base + 0.1 * x + np.random.normal(0, noise) for x in x_values]
            samples[sample_name] = {x: y for x, y in zip(x_values, y_values)}
        return samples

    def _create_plots(self, all_groups):
        """Create three side-by-side plots with single shared dropdown"""

        # Separate data by plot position
        plot1_data = [group[0] for group in all_groups]
        plot2_data = [group[1] for group in all_groups]
        plot3_data = [group[2] for group in all_groups]

        # Create data labels for all 10 groups
        data_labels = [{"name": f"Experiment {i}", "ylab": f"Value (Group {i})"} for i in range(1, 11)]

        # Plot configurations - note we're hiding the default buttons
        plot1_config = {
            "id": "sync_plot_1",
            "title": "Metric A",
            "ylab": "Value",
            "xlab": "Position",
            "data_labels": data_labels,
            "hide_buttons": True,  # Hide individual plot buttons
        }

        plot2_config = {
            "id": "sync_plot_2",
            "title": "Metric B",
            "ylab": "Value",
            "xlab": "Position",
            "data_labels": data_labels,
            "hide_buttons": True,
        }

        plot3_config = {
            "id": "sync_plot_3",
            "title": "Metric C",
            "ylab": "Value",
            "xlab": "Position",
            "data_labels": data_labels,
            "hide_buttons": True,
        }

        # Generate the plots
        plot1_html = linegraph.plot(plot1_data, plot1_config).add_to_report(
            module_anchor="example_module", section_anchor="multiplot_comparison"
        )
        plot2_html = linegraph.plot(plot2_data, plot2_config).add_to_report(
            module_anchor="example_module", section_anchor="multiplot_comparison"
        )
        plot3_html = linegraph.plot(plot3_data, plot3_config).add_to_report(
            module_anchor="example_module", section_anchor="multiplot_comparison"
        )

        # Create a single dropdown that controls all three plots
        dropdown_options = "".join(
            [f'<option value="{i}">{label["name"]}</option>' for i, label in enumerate(data_labels)]
        )

        # Combine plots with single shared dropdown
        combined_html = f"""
        <div id="sync_plots_container">
            <div style="margin-bottom: 15px;">
                <label for="sync_plot_selector" style="font-weight: bold; margin-right: 10px;">
                    Select Experiment:
                </label>
                <select id="sync_plot_selector" class="form-control" style="display: inline-block; width: auto;">
                    {dropdown_options}
                </select>
            </div>
            
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px;">
                    {plot1_html}
                </div>
                <div style="flex: 1; min-width: 300px;">
                    {plot2_html}
                </div>
                <div style="flex: 1; min-width: 300px;">
                    {plot3_html}
                </div>
            </div>
        </div>
        
        <script type="text/javascript">
        $(function() {{
            // Function to switch all three plots using MultiQC's internal method
            function switchSyncedPlots(datasetIndex) {{

                console.log("Switching all plots to dataset index:", datasetIndex);
                var plotIds = ['#sync_plot_1', '#sync_plot_2', '#sync_plot_3'];
                
                plotIds.forEach(function(plotId) {{
                    if (window.mqc_plots && window.mqc_plots[plotId]) {{
                        var plot = window.mqc_plots[plotId];
                        var oldIdx = plot.activeDatasetIdx;
                        plot.activeDatasetIdx = datasetIndex;
                        
                        // Only re-render if dataset changed and plot is already rendered
                        if (oldIdx !== datasetIndex && plot.rendered) {{
                            // Use MultiQC's internal switch function
                            if (typeof window.$s === 'function') {{
                                window.$s(plotId);
                            }}
                        }}
                    }}
                }});
            }}
            
            // Handle dropdown change
            $('#sync_plot_selector').on('change', function() {{
                var selectedIndex = parseInt($(this).val());
                switchSyncedPlots(selectedIndex);
            }});
        }});
        </script>
        """

        # Add section to report
        self.add_section(
            name="Multi-Plot Comparison",
            anchor="multiplot_comparison",
            description="Three related plots controlled by a single dropdown selector",
            plot=combined_html,
        )
