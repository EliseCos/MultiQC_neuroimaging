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

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Harmonization distribution results",
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
        bundles = list(self.harmonized_df["roi"].unique())[:2]
        headers = set(self.harmonized_df.columns)
        meta_columns = set(["site", "age", "sex", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        metric_columns = headers - non_metric_columns
        # Filter metric_columns to keep only "fa, md, rd columns"
        metric_columns = [m for m in metric_columns if m in ["fa", "md"]]

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
                # assert bundle in self.harmonization_json.data, \
                #     f"The bundle {bundle} is missing from the harmonization JSON."
                # assert metric in self.harmonization_json.data[bundle], \
                #     f"The metric {metric} is missing from the harmonization JSON for bundle {bundle}."

        is_first_tuple = True
        all_div_ids = []
        anchors = {}
        plots_html_content = ""
        for bundle in bundles:
            for metric in metric_columns:
                log.info(f"Processing bundle: {bundle}, metric: {metric}")
                html, plot_anchors, div_id = self.make_flex_row(bundle, metric, hidden=not is_first_tuple)

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

    def make_flex_row(self, bundle, metric, hidden=False):
        div_id = f"{bundle.replace(' ', '-')}_{metric.replace(' ', '-')}"
        # Generate the 3 individual plots
        # Note: Use unique IDs to prevent DOM conflicts!
        plot_anchors = [f"p1_{div_id}", f"p2_{div_id}", f"p3_{div_id}"]

        p1_html, p1_anchor = self.create_datamodel_figure(bundle, metric)
        p2_html, p2_anchor = self.create_agecurve_raw_figure(bundle, metric)
        p3_html, p3_anchor = self.create_agecurve_harmonized_figure(bundle, metric)
        # p4_html, p4_anchor = self.create_battacharia_figure(bundle, metric)

        plot_anchors = [p1_anchor, p2_anchor, p3_anchor]

        return (
            f"""
        <div id="{div_id}" style="width: 90%; display: {"none" if hidden else "flex"}; flex-direction: row; justify-content: space-between;">
            <div style="width:100%; margin: 0 auto;">{p1_html}</div>
            <div style="width:100%; margin: 0 auto;">{p2_html}</div>
            <div style="width:100%; margin: 0 auto;">{p3_html}</div>
        </div>
        """,
            {"plot_anchors": plot_anchors, "div_id": div_id},
            div_id,
        )

    def _get_data(self, bundle, metric, disease=None) -> Dict[str, Any]:
        """Get example data for plotting."""
        data = self.rois_metrics_data[bundle][metric]
        if disease:
            # Filter data for the specified disease
            indices = [i for i, s in enumerate(data["samples"]) if self.meta[s]["disease"] == disease]

            data["samples"] = [data["samples"][i] for i in indices]
            data["ages"] = [data["ages"][i] for i in indices]
            data["values"] = [data["values"][i] for i in indices]

        return data
    
    ########################################
    # DataModel plot
    ########################################
    def create_datamodel_figure(self, bundle, metric):
        fig = go.Figure()

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
            )
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
            )
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
            )
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
            )
        )

        fig.update_layout(
            title_text="Data Model",
            title_subtitle_text=f"Bundle {bundle} | Metric: {metric}",
            title_x=0.5,
            xaxis_title_text="Age",
            yaxis_title_text=metric,
            legend_yanchor="auto",
            legend_xanchor="auto",
            legend_x=0,
            legend_y=0
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"datamodel_{bundle}_{metric}"
    
    ########################################
    # AgeCurve Raw plot
    ########################################
    def create_agecurve_raw_figure(self, bundle, metric):
        """Create a AgeCurve Raw figure to visualize the data."""
        fig = go.Figure()

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
                    legendgroup="reference_percentiles",
                    showlegend=False,
                    opacity=0.2
                )
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
                    legendgroup="moving_percentiles",
                    showlegend=False,
                    opacity=0.2
                )
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
                legendgroup="reference_percentiles",
                line=dict(color=reference_percentile_plots[ref_median_curve]["color"], width=4),
            )
        )

        mov_median_curve = mov_available_percentiles[mov_middle]
        fig.add_trace(
            go.Scatter(
                x=moving_percentile_plots[mov_median_curve]['data_x'],
                y=moving_percentile_plots[mov_median_curve]['data_y'],
                mode="lines",
                name="Moving",
                legendgroup="moving_percentiles",
                line=dict(color=moving_percentile_plots[mov_median_curve]["color"], width=4),
            )
        )

        fig.update_layout(
            title_text="Pre-Harmonization",
            title_subtitle_text=f"Bundle {bundle} | Metric: {metric}",
            title_x=0.5,
            xaxis_title_text="Age",
            yaxis_title_text=metric,
            legend_yanchor="auto",
            legend_xanchor="auto",
            legend_x=0,
            legend_y=0
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"age_curve_raw_{bundle}_{metric}"

    ########################################
    # AgeCurve Harmonized plot
    ########################################
    def create_agecurve_harmonized_figure(self, bundle, metric):
        """Create a AgeCurve Harmonized figure to visualize the data."""
        fig = go.Figure()

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
                    legendgroup="reference_percentiles",
                    showlegend=False,
                    opacity=0.2
                )
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
                    legendgroup="moving_percentiles",
                    showlegend=False,
                    opacity=0.2
                )
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
                legendgroup="reference_percentiles",
                line=dict(color=reference_percentile_plots[ref_median_curve]["color"], width=4),
            )
        )

        mov_median_curve = mov_available_percentiles[mov_middle]
        fig.add_trace(
            go.Scatter(
                x=moving_percentile_plots[mov_median_curve]['data_x'],
                y=moving_percentile_plots[mov_median_curve]['data_y'],
                mode="lines",
                name="Moving",
                legendgroup="moving_percentiles",
                line=dict(color=moving_percentile_plots[mov_median_curve]["color"], width=4),
            )
        )

        fig.update_layout(
            title_text="Post-Harmonization",
            title_subtitle_text=f"Bundle {bundle} | Metric: {metric}",
            title_x=0.5,
            xaxis_title_text="Age",
            yaxis_title_text=metric,
            legend_yanchor="auto",
            legend_xanchor="auto",
            legend_x=0,
            legend_y=0
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"agecurve_harmonized_{bundle}_{metric}"
