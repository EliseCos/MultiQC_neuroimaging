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
import pandas as pd

from .utils import load_sites_data

from .combat.plotjson import PlotJson, PlotJsonAggregator

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")

class MultiqcModule(BaseMultiqcModule):
    """
    This section aims to visualize the result of the harmonization_plotly process.
    """

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Harmonization distribution results",
            anchor="harmonization_plotly",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=self.__doc__,
        )
        log.info
        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Find files using the custom search pattern added in custom_code
        files = list(self.find_log_files("harmonization_plotly"))

        tsv_files = []
        json_files = []
        unknown_files = []
        for f in files:
            if f["fn"].endswith(".tsv"):
                tsv_files.append(f)
            elif f["fn"].endswith(".json"):
                json_files.append(f)
            else:
                unknown_files.append(f)

        if len(unknown_files) > 0:
            log.warning(
                f"Found {len(unknown_files)} files with unknown extension for harmonization_plotly module. "
                f"Only .tsv and .json files are supported. Files: {[f['fn'] for f in unknown_files]}"
            )

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(files) == 0:
            log.debug(f"Could not find harmonization_plotly reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        self.df = load_sites_data(tsv_files)
        
        # Read plot data from JSON files and aggregate
        self.plots_json = PlotJsonAggregator()
        for jf in json_files:
            pa = PlotJsonAggregator.from_json(jf["f"])
            self.plots_json.add_plot_json_aggregator(pa)
        
        # Write the aggregated JSON file for reference
        aggregated_json_path = os.path.join(config.output_dir, "harmonization_plotly_aggregated_plots.json")
        with open(aggregated_json_path, 'w') as f:
            json.dump(self.plots_json.aggregated_data, f, indent=4)
        log.info(f"Aggregated plot JSON saved to {aggregated_json_path}")

        # Extract list of metrics and non-metric columns
        bundles = list(self.df["roi"].unique())
        headers = set(self.df.columns)
        meta_columns = set(["site", "age", "sex", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        metric_columns = headers - non_metric_columns
        # Filter metric_columns to keep only "fa, md, rd columns"
        metric_columns = [m for m in metric_columns if m in ["fa", "md", "rd"]]

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

        plot_anchors = [p1_anchor, p2_anchor, p3_anchor]

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

        data = self.df[self.df["roi"] == bundle]

        # Add scatterplot of data
        fig.add_trace(
            go.Scatter(
                x=data["age"],
                y=data[metric],
                mode="markers",
                text=data["sample"]
            )
        )

        # Add reference regression line which was calculated by Clinical-Combat
        plot_json = self.plots_json.data[bundle][metric]
        ref_y = plot_json["regression_reference"]["data_y"]
        fig.add_trace(
            go.Scatter(
                x=data["age"],
                y=ref_y,
                mode="lines",
                name=f"Reference ({plot_json['regression_reference']['site']})",
                line=dict(color="red", dash="dash"),
            )
        )

        mov_y = plot_json["regression_moving"]["data_y"]
        fig.add_trace(
            go.Scatter(
                x=data["age"],
                y=mov_y,
                mode="lines",
                name=f"Moving ({plot_json['regression_moving']['site']}))",
                line=dict(color="blue", dash="dash"),
            )
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"datamodel_{bundle}_{metric}"
    
    ########################################
    # AgeCurve Raw plot
    ########################################
    def create_agecurve_raw_figure(self, bundle, metric):
        """Create a AgeCurve Raw figure to visualize the data."""
        fig = go.Figure()

        data = self.df[self.df["roi"] == bundle]

        # Add scatterplot of data
        fig.add_trace(
            go.Scatter(
                x=data["age"],
                y=data[metric],
                mode="markers",
                text=data["sample"]
            )
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"age_curve_raw_{bundle}_{metric}"

    ########################################
    # AgeCurve Harmonized plot
    ########################################
    def create_agecurve_harmonized_figure(self, bundle, metric):
        """Create a AgeCurve Harmonized figure to visualize the data."""
        fig = go.Figure()

        data = self.df[self.df["roi"] == bundle]

        # Add scatterplot of data
        fig.add_trace(
            go.Scatter(
                x=data["age"],
                y=data[metric],
                mode="markers",
                text=data["sample"]
            )
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False), f"agecurve_harmonized_{bundle}_{metric}"
