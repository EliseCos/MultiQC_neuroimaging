"""MultiQC module for harmonization results."""

import os
import csv
import logging
from typing import Dict, Any
from collections import defaultdict
import itertools

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
import pandas as pd

from .utils import load_sites_data, PALETTE, to_plotly_rgb

from .combat.plotjson import PlotJsonAggregator

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")

class MultiqcModule(BaseMultiqcModule):
    """
    This section aims to visualize the result of the harmonization_plotly process.
    """
    REF_STATS_SP_KEY            = "harmonization_plotly/reference_stats"
    RAW_STATS_SP_KEY            = "harmonization_plotly/raw_stats"
    HARMONIZED_STATS_SP_KEY     = "harmonization_plotly/harmonized_stats"
    DATA_MODELS_PLOTS_SP_KEY    = "harmonization_plotly/data_models_plots"
    HARMONIZATION_PLOTS_SP_KEY  = "harmonization_plotly/harmonization_plots"
    HARMONIZATION_DISTANCE_SP_KEY  = "harmonization_plotly/harmonization_distance"

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Harmonization results",
            anchor="harmonization_plotly",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=self.__doc__,
        )

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
            log.debug(f"Could not find harmonization_plotly reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        self.ref_df        = load_sites_data(ref_stats_file)
        self.raw_df        = load_sites_data(raw_stats_files)
        self.harmonized_df = load_sites_data(harmonized_stats_files)
        
        # Read plot data from JSON files and aggregate
        self.data_model_json    = PlotJsonAggregator.from_json(list(map(lambda f: f["f"], data_model_plots_files)))
        self.harmonization_json = PlotJsonAggregator.from_json(list(map(lambda f: f["f"], harmonization_plots_files)))
        
        # Write the aggregated JSON file for reference
        aggregated_json_path = os.path.join(config.output_dir, "datamodels_aggregated.json")
        with open(aggregated_json_path, 'w') as f:
            json.dump(self.data_model_json.aggregated_data, f, indent=4)
        log.info(f"Aggregated plot JSON saved to {aggregated_json_path}")

        aggregated_json_path = os.path.join(config.output_dir, "harmonization_aggregated.json")
        with open(aggregated_json_path, 'w') as f:
            json.dump(self.harmonization_json.aggregated_data, f, indent=4)
        log.info(f"Aggregated plot JSON saved to {aggregated_json_path}")

        # Extract list of metrics and non-metric columns
        bundles = list(self.harmonized_df["roi"].unique())
        headers = set(self.harmonized_df.columns)
        meta_columns = set(["site", "age", "sex", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        metric_columns = headers - non_metric_columns
        # Filter metric_columns to keep only "fa, md, rd columns"
        metric_columns = [m for m in metric_columns if m in ["fa", "md", "rd", "ad"]]

        # Verify that the raw_df and harmonized_df have the same samples
        assert set(self.raw_df["sample"]) == set(self.harmonized_df["sample"]), \
            "The raw and harmonized data files must contain the same samples."
        
        # Verify that the raw and harmonized data have the same ROIs
        assert set(self.raw_df["roi"]) == set(self.harmonized_df["roi"]), \
            "The raw and harmonized data files must contain the same ROIs."
        
        # Verify that the dataframes and the jsons have the same ROIs and metrics
        for bundle in bundles:
            for metric in metric_columns:
                assert bundle in self.data_model_json.data, \
                    f"The bundle {bundle} is missing from the data model JSON."
                assert metric in self.data_model_json.data[bundle], \
                    f"The metric {metric} is missing from the data model JSON for bundle {bundle}."
                assert bundle in self.harmonization_json.data, \
                    f"The bundle {bundle} is missing from the harmonization JSON."
                assert metric in self.harmonization_json.data[bundle], \
                    f"The metric {metric} is missing from the harmonization JSON for bundle {bundle}."

        is_first_tuple = True
        all_div_ids = []
        anchors = {}
        plots_html_content = ""
        for bundle in bundles:
            for metric in metric_columns:
                log.info(f"Processing bundle: {bundle}, metric: {metric}")
                html, div_id = self.make_flex_row(bundle, metric, hidden=not is_first_tuple)

                plots_html_content += html

                # Keep the div IDs and anchors for JS switching
                all_div_ids.append(div_id)

                if bundle not in anchors:
                    anchors[bundle] = {}
                anchors[bundle][metric] = div_id
                is_first_tuple = False

        html_bundle_options = "".join(
            [
                f'<option value="{bundle}" {"selected" if i == 0 else ""}>{bundle}</option>'
                for i, bundle in enumerate(bundles)
            ]
        )
        html_metric_options = "".join(
            [
                f'<option value="{metric}" {"selected" if i == 0 else ""}>{metric}</option>'
                for i, metric in enumerate(metric_columns)
            ]
        )
        default_bundle = list(bundles)[0]
        default_metric = list(metric_columns)[0]
        html_script = f"""
        <div class="d-flex justify-content-center gap-4">
            <div class="text-center">
                <label class="form-label mb-2 fw-semibold">Bundle</label>
                <select class="form-select shadow-sm" 
                        aria-label="Bundle selection" 
                        onchange="showScenario(this.value, current_metric)"
                        style="min-width: 150px;">
                    {html_bundle_options}
                </select>
            </div>

            <div class="text-center">
                <label class="form-label mb-2 fw-semibold">Metric</label>
                <select class="form-select shadow-sm" 
                        aria-label="Metric selection" 
                        onchange="showScenario(current_bundle, this.value)"
                        style="min-width: 150px;">
                    {html_metric_options}
                </select>
            </div>
        </div>
        <script>
        var div_ids = {json.dumps(all_div_ids)};
        var anchors = {json.dumps(anchors)};
        var current_bundle = "{default_bundle}";
        var current_metric = "{default_metric}";

        function renderPlotlyPlots(parentDiv) {{
            // Find ALL plotly graph divs inside it and resize each
            setTimeout(() => {{
                parentDiv.querySelectorAll('.plotly-graph-div').forEach(plotDiv => {{
                    Plotly.relayout(plotDiv, {{}});
                }});
            }}, 0);  // Even 0ms helps by deferring to next event loop
        }}

        function showScenario(bundle, metric) {{
            console.log("Switching to view: ", bundle, ",", metric);
            var metric_changed = (metric !== current_metric);
            current_bundle = bundle;
            current_metric = metric;

            // Make sure all plot divs are hidden before showing the selected one
            div_ids.forEach(function(divId) {{
                document.getElementById(divId).style.display = 'none';
            }});
            
            // Show the selected scenario
            var selected_div_id = anchors[bundle][metric];
            console.log("Showing div ID:", selected_div_id);
            document.getElementById(selected_div_id).style.display = 'block';

            const subPlotsParentDiv = document.getElementById(selected_div_id);
            renderPlotlyPlots(subPlotsParentDiv);

            // Also render the bhattacharyya_plot_div plot if the metric is changed
            if (metric_changed) {{
                renderBhattacharyyaPlots(metric);
            }}
        }}
        </script>
        """

        # Add the section to the report
        html_content = html_script + plots_html_content
        self.add_section(
            name="Distributional results",
            anchor="harmonization_distributional_results",
            content=html_content)
        
        # Load Bhattacharyya distance data
        # In theory, there should be two categories: "raw" and "harmonized"
        # The pre-harmonization files should be named: Site.metric.raw.bhattacharyya.txt
        # The post-harmonization files should be named: Site.metric.clinical.harmonized.bhattacharyya.txt
        # Each file as two lines with values separated by white spaces.
        # The first line of the file contains the ROIs/bundles names.
        # The second line contains the corresponding Bhattacharyya distance values.
        # Exception: for some reason, the first column contains the subject/sample count
        # of healthly controls for that site, so we will discard that column as we have no
        # use for it at the moment.
        bhattacharyya_html_content = ""
        bhattacharyya_data = {"raw": defaultdict(lambda: defaultdict(list)), "harmonized": defaultdict(lambda: defaultdict(list))}
        all_metrics = set()
        for f in harmonization_distance_files:
            filename = f["fn"]
            lines = f["f"].strip().split("\n")

            if len(lines) < 2:
                raise ValueError(f"Bhattacharyya distance file {filename} must contain at least two lines.")
            
            bundles     = lines[0].strip().split()[1:]  # Skip first column (sample count)
            distances   = lines[1].strip().split()[1:]  # Skip first column (sample count)

            if len(bundles) != len(distances):
                raise ValueError(f"Bhattacharyya distance file {filename} has mismatched number of bundles and distances.")
            
            # Extract site, metric, and harmonization status from filename
            parts = os.path.basename(filename).split(".")
            if len(parts) < 4:
                raise ValueError(f"Bhattacharyya distance filename {filename} is not in the expected format.")
            
            metric = parts[1]
            all_metrics.add(metric)
            status = "harmonized" if "harmonized" in parts else "raw"

            for b, d in zip(bundles, distances):
                bhattacharyya_data[status][metric]["bundles"].append(b)
                bhattacharyya_data[status][metric]["distances"].append(float(d))

        # Create the boxplot using Plotly
        all_metrics = bhattacharyya_data["raw"].keys() | bhattacharyya_data["harmonized"].keys()
        b_div_ids = {}
        for metric in all_metrics:
            fig = go.Figure()

            # Raw data
            if metric in bhattacharyya_data["raw"]:
                fig.add_trace(
                    go.Box(
                        y=bhattacharyya_data["raw"][metric]["distances"],
                        name="Pre-Harmonization",
                        boxmean=True,
                        marker_color="darkblue",
                        text=bhattacharyya_data["raw"][metric]["bundles"]
                    )
                )

            # Harmonized data
            if metric in bhattacharyya_data["harmonized"]:
                fig.add_trace(
                    go.Box(
                        y=bhattacharyya_data["harmonized"][metric]["distances"],
                        name="Post-Harmonization",
                        marker_color="royalblue",
                        text=bhattacharyya_data["harmonized"][metric]["bundles"],
                        boxmean=True
                    )
                )

            fig.update_layout(
                title=f"Mean Bhattacharyya distance across bundles<br>Metric: {metric}",
                title_xanchor="center",
                title_x=0.5,
                yaxis_title="Bhattacharyya Distance",
                height=500,
                legend_visible=False,
                
            )

            b_div_id = f"bhattacharyya_{metric.replace(' ', '-')}"
            b_div_ids[metric] = b_div_id

            bhattacharyya_html_content += f"""
            <div id="{b_div_id}" style="max-width:800px; margin: 0 auto; display: {"none" if metric != default_metric else "block"};">
                {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
            </div>
            """
        bhattacharyya_html_script = f"""
        <script>
        var bhattacharyya_div_ids = {json.dumps(b_div_ids)};
        function renderBhattacharyyaPlots(metric) {{
            console.log("Rendering Bhattacharyya plot for metric:", metric);
            // Make every plot div hidden before showing the selected one
            Object.values(bhattacharyya_div_ids).forEach(function(divId) {{
                document.getElementById(divId).style.display = 'none';
            }});

            // Find ALL plotly graph divs inside it and resize each
            var divId = bhattacharyya_div_ids[metric];
            document.getElementById(divId).style.display = 'block';
            console.log("Rendering div ID:", divId);
            renderPlotlyPlots(document.getElementById(divId));
        }}
        </script>
        """

        self.add_section(
            name="Mean Bhattacharyya Distance (BD)",
            anchor="bhattacharyya_distance",
            content=bhattacharyya_html_script + bhattacharyya_html_content)
        

    def make_flex_row(self, bundle, metric, hidden=False):
        div_id = f"{bundle.replace(' ', '-')}_{metric.replace(' ', '-')}"
        # Generate the 3 individual plots
        # Note: Use unique IDs to prevent DOM conflicts!
        
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=True,
            horizontal_spacing=0.025,
            subplot_titles=("Data Model", "Pre-Harmonization", "Post-Harmonization")
        )

        self.create_datamodel_figure(bundle, metric, fig=fig, row=1, col=1)
        self.create_agecurve_raw_figure(bundle, metric, fig=fig, row=1, col=2)
        self.create_agecurve_harmonized_figure(bundle, metric, fig=fig, row=1, col=3)

        fig.update_layout(
            autosize=True,
            title_text=f"Bundle: {bundle} | Metric: {metric}",
            title_xanchor="center",
            title_x=0.5,
            legend_orientation="h",
            legend_y=-0.15,
            legend_xanchor="center",
            legend_x=0.5,
            margin_b=40,
            height=500
        )

        fig.update_xaxes(title_text="Age")
        fig.update_yaxes(title_text=metric.upper(), row=1, col=1)

        return (
            f"""
        <div id="{div_id}" style="width:100%; display: {"none" if hidden else "block"};">
            {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
        </div>
        """,
        div_id,
        )
    
    ########################################
    # DataModel plot
    ########################################
    def create_datamodel_figure(self, bundle, metric, fig, row, col):
        ref_data = self.ref_df[self.ref_df["roi"] == bundle]
        raw_data = self.raw_df[self.raw_df["roi"] == bundle]

        plot_json = self.data_model_json.data[bundle][metric]
        ref_color = to_plotly_rgb(plot_json["regression_reference"]["color"]) \
            if "regression_reference" in plot_json and "color" in plot_json["regression_reference"] else PALETTE[0]
        moving_color = to_plotly_rgb(plot_json["regression_moving"]["color"]) \
            if "regression_moving" in plot_json and "color" in plot_json["regression_moving"] else PALETTE[1]
        
        reference_name = f"{plot_json['regression_reference']['site']} (reference)"
        moving_name = f"{plot_json['regression_moving']['site']} (moving)"

        # Add reference scatterplot of data
        fig.add_trace(
            go.Scatter(
                x=ref_data["age"],
                y=ref_data[metric],
                mode="markers",
                text=ref_data["sample"],
                marker=dict(color=ref_color),
                name=reference_name,
                legendgroup="reference_group"
            ),
            row=row,
            col=col
        )

        # Add (raw) moving scatter
        fig.add_trace(
            go.Scatter(
                x=raw_data["age"],
                y=raw_data[metric],
                mode="markers",
                text=raw_data["sample"],
                marker=dict(color=moving_color),
                name=moving_name,
                legendgroup="moving_group"
            ),
            row=row,
            col=col
        )

        # Add reference regression line which was calculated by Clinical-Combat
        ref_x = plot_json["regression_reference"]["data_x"]
        ref_y = plot_json["regression_reference"]["data_y"]
        fig.add_trace(
            go.Scatter(
                x=ref_x,
                y=ref_y,
                mode="lines",
                name=reference_name + " - regression",
                line=dict(color=ref_color),
                legendgroup="reference_group",
                showlegend=False
            ),
            row=row,
            col=col
        )

        mov_x = plot_json["regression_moving"]["data_x"]
        mov_y = plot_json["regression_moving"]["data_y"]
        fig.add_trace(
            go.Scatter(
                x=mov_x,
                y=mov_y,
                mode="lines",
                name=moving_name + " - regression",
                line=dict(color=moving_color),
                legendgroup="moving_group",
                showlegend=False
            ),
            row=row,
            col=col
        )
    
    ########################################
    # AgeCurve Raw plot
    ########################################
    def create_agecurve_raw_figure(self, bundle, metric, fig, row, col):
        """Create a AgeCurve Raw figure to visualize the data."""

        plot_json = self.harmonization_json.data[bundle][metric]

        ############################
        # Reference percentiles
        ############################
        reference_percentile_plots = plot_json['reference_percentiles']

        # Extract the percentile curves dict keys ordered by percentile number.
        ref_percentile_plots = [k for k in reference_percentile_plots.keys()]
        ref_percentiles = [v["percentile"] for v in reference_percentile_plots.values()]
        ref_available_percentiles = [p for _, p in sorted(zip(ref_percentiles, ref_percentile_plots))]

        # Make sure the number of percentiles is odd to have a middle one
        if len(ref_available_percentiles) % 2 == 0:
            raise ValueError("The number of percentiles must be odd to have a main/mean percentile.")

        ref_middle = len(ref_available_percentiles) // 2
        ref_other_percentiles = ref_available_percentiles[:ref_middle] + ref_available_percentiles[ref_middle+1:]

        # Make sure the 5th percentile is matched with the 95th, 10th with 90th, etc.
        for i in range(len(ref_other_percentiles)//2):
            p1_name = ref_other_percentiles[i]
            p2_name = ref_other_percentiles[-(i+1)]

            p1 = reference_percentile_plots[p1_name]
            p2 = reference_percentile_plots[p2_name]
            
            # We need to fill the area between p1 and p2
            fig.add_trace(
                go.Scatter(
                    x=p1['data_x'] + p2['data_x'][::-1],
                    y=p1['data_y'] + p2['data_y'][::-1],
                    fill='toself',
                    fillcolor=p1["color"],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"Reference {p1['percentile']}th - {p2['percentile']}th percentile",
                    legendgroup="reference_group",
                    showlegend=False,
                    opacity=0.2
                ),
                row=row,
                col=col
            )

        ############################
        # Moving percentiles
        ############################
        moving_percentile_plots = plot_json['moving_raw_percentiles']

        # Extract the percentile curves dict keys ordered by percentile number.
        mov_percentile_plots = [k for k in moving_percentile_plots.keys()]
        mov_percentiles = [v["percentile"] for v in moving_percentile_plots.values()]
        mov_available_percentiles = [p for _, p in sorted(zip(mov_percentiles, mov_percentile_plots))]

        # Make sure the number of percentiles is odd to have a middle one
        if len(mov_available_percentiles) % 2 == 0:
            raise ValueError("The number of percentiles must be odd to have a main/mean percentile.")

        mov_middle = len(mov_available_percentiles) // 2

        mov_other_percentiles = mov_available_percentiles[:mov_middle] + mov_available_percentiles[mov_middle+1:]

        # Make sure the 5th percentile is matched with the 95th, 10th with 90th, etc.
        for i in range(len(mov_other_percentiles)//2):
            p1_name = mov_other_percentiles[i]
            p2_name = mov_other_percentiles[-(i+1)]

            p1 = moving_percentile_plots[p1_name]
            p2 = moving_percentile_plots[p2_name]
            
            # We need to fill the area between p1 and p2
            fig.add_trace(
                go.Scatter(
                    x=p1['data_x'] + p2['data_x'][::-1],
                    y=p1['data_y'] + p2['data_y'][::-1],
                    fill='toself',
                    fillcolor=p1["color"],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"Moving {p1['percentile']}th - {p2['percentile']}th percentile",
                    legendgroup="moving_group",
                    showlegend=False,
                    opacity=0.2
                ),
                row=row,
                col=col
            )


        ############################
        # Median curves
        ############################
        # Plot the median curves last so they get drawn on top of the filled areas.
        ref_median_curve = ref_available_percentiles[ref_middle]
        fig.add_trace(
            go.Scatter(
                x=reference_percentile_plots[ref_median_curve]['data_x'],
                y=reference_percentile_plots[ref_median_curve]['data_y'],
                mode="lines",
                name="Reference",
                legendgroup="reference_group",
                showlegend=False,
                line=dict(color=reference_percentile_plots[ref_median_curve]["color"], width=4),
            ),
            row=row,
            col=col
        )

        mov_median_curve = mov_available_percentiles[mov_middle]
        fig.add_trace(
            go.Scatter(
                x=moving_percentile_plots[mov_median_curve]['data_x'],
                y=moving_percentile_plots[mov_median_curve]['data_y'],
                mode="lines",
                name="Moving",
                legendgroup="moving_group",
                showlegend=False,
                line=dict(color=moving_percentile_plots[mov_median_curve]["color"], width=4),
            ),
            row=row,
            col=col
        )

    ########################################
    # AgeCurve Harmonized plot
    ########################################
    def create_agecurve_harmonized_figure(self, bundle, metric, fig, row, col):
        """Create a AgeCurve Harmonized figure to visualize the data."""

        plot_json = self.harmonization_json.data[bundle][metric]

        ############################
        # Reference percentiles
        ############################
        reference_percentile_plots = plot_json['reference_percentiles']

        # Extract the percentile curves dict keys ordered by percentile number.
        ref_percentile_plots = [k for k in reference_percentile_plots.keys()]
        ref_percentiles = [v["percentile"] for v in reference_percentile_plots.values()]
        ref_available_percentiles = [p for _, p in sorted(zip(ref_percentiles, ref_percentile_plots))]

        # Make sure the number of percentiles is odd to have a middle one
        if len(ref_available_percentiles) % 2 == 0:
            raise ValueError("The number of percentiles must be odd to have a main/mean percentile.")

        ref_middle = len(ref_available_percentiles) // 2
        ref_other_percentiles = ref_available_percentiles[:ref_middle] + ref_available_percentiles[ref_middle+1:]

        # Make sure the 5th percentile is matched with the 95th, 10th with 90th, etc.
        for i in range(len(ref_other_percentiles)//2):
            p1_name = ref_other_percentiles[i]
            p2_name = ref_other_percentiles[-(i+1)]

            p1 = reference_percentile_plots[p1_name]
            p2 = reference_percentile_plots[p2_name]
            
            # We need to fill the area between p1 and p2
            fig.add_trace(
                go.Scatter(
                    x=p1['data_x'] + p2['data_x'][::-1],
                    y=p1['data_y'] + p2['data_y'][::-1],
                    fill='toself',
                    fillcolor=p1["color"],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"Reference {p1['percentile']}th - {p2['percentile']}th percentile",
                    legendgroup="reference_group",
                    showlegend=False,
                    opacity=0.2
                ),
                row=row,
                col=col
            )

        ############################
        # Moving percentiles
        ############################
        moving_percentile_plots = plot_json['moving_harmonized_percentiles']

        # Extract the percentile curves dict keys ordered by percentile number.
        mov_percentile_plots = [k for k in moving_percentile_plots.keys()]
        mov_percentiles = [v["percentile"] for v in moving_percentile_plots.values()]
        mov_available_percentiles = [p for _, p in sorted(zip(mov_percentiles, mov_percentile_plots))]

        # Make sure the number of percentiles is odd to have a middle one
        if len(mov_available_percentiles) % 2 == 0:
            raise ValueError("The number of percentiles must be odd to have a main/mean percentile.")

        mov_middle = len(mov_available_percentiles) // 2

        mov_other_percentiles = mov_available_percentiles[:mov_middle] + mov_available_percentiles[mov_middle+1:]

        # Make sure the 5th percentile is matched with the 95th, 10th with 90th, etc.
        for i in range(len(mov_other_percentiles)//2):
            p1_name = mov_other_percentiles[i]
            p2_name = mov_other_percentiles[-(i+1)]

            p1 = moving_percentile_plots[p1_name]
            p2 = moving_percentile_plots[p2_name]
            
            # We need to fill the area between p1 and p2
            fig.add_trace(
                go.Scatter(
                    x=p1['data_x'] + p2['data_x'][::-1],
                    y=p1['data_y'] + p2['data_y'][::-1],
                    fill='toself',
                    fillcolor=p1["color"],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"Moving {p1['percentile']}th - {p2['percentile']}th percentile",
                    legendgroup="moving_group",
                    showlegend=False,
                    opacity=0.2
                ),
                row=row,
                col=col
            )


        ############################
        # Median curves
        ############################
        # Plot the median curves last so they get drawn on top of the filled areas.
        ref_median_curve = ref_available_percentiles[ref_middle]
        fig.add_trace(
            go.Scatter(
                x=reference_percentile_plots[ref_median_curve]['data_x'],
                y=reference_percentile_plots[ref_median_curve]['data_y'],
                mode="lines",
                name="Reference",
                legendgroup="reference_group",
                showlegend=False,
                line=dict(color=reference_percentile_plots[ref_median_curve]["color"], width=4),
            ),
            row=row,
            col=col
        )

        mov_median_curve = mov_available_percentiles[mov_middle]
        fig.add_trace(
            go.Scatter(
                x=moving_percentile_plots[mov_median_curve]['data_x'],
                y=moving_percentile_plots[mov_median_curve]['data_y'],
                mode="lines",
                name="Moving",
                legendgroup="moving_group",
                showlegend=False,
                line=dict(color=moving_percentile_plots[mov_median_curve]["color"], width=4),
            ),
            row=row,
            col=col
        )
