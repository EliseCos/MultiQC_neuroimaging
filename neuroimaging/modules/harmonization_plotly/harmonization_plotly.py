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
from io import StringIO

from .utils import load_sites_data

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")


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

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(files) == 0:
            log.debug(f"Could not find harmonization_plotly reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        df = load_sites_data(files)

        # --- 3. Generate content for each tab ---
        is_first_tuple = True
        all_div_ids = []
        anchors = {}
        plots_html_content = ""
        for bundle, metrics in df.items():
            for metric, data in metrics.items():
                html, plot_anchors, div_id = self.make_flex_row(data, bundle, metric, hidden=not is_first_tuple)

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

    def make_flex_row(self, plot_data_list, bundle, metric, hidden=False):
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

    def create_datamodel_figure(self, bundle, metric):
        """Create a DataModel figure to visualize the data."""
        logging.info(f"Creating DataModel figure for bundle: {bundle}, metric: {metric}")
        # data = self._get_data(bundle, metric, "HC")
        pass

    def create_agecurve_raw_figure(self, bundle, metric):
        """Create a AgeCurve Raw figure to visualize the data."""
        pass

    def create_agecurve_harmonized_figure(self, bundle, metric):
        """Create a AgeCurve Harmonized figure to visualize the data."""
        pass

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
                data[roi][metric][sample] = value

        # Wrote the meta to a meta.json file
        with open("C:\\Users\\jerem\\Documents\\multiqc\\MultiQC_neuroimaging\\meta.json", "w") as meta_file:
            json.dump(dict(meta), meta_file, indent=4)

        # Wrote the data to a data.json file
        with open("C:\\Users\\jerem\\Documents\\multiqc\\MultiQC_neuroimaging\\data.json", "w") as data_file:
            json.dump(dict(data), data_file, indent=4)

        rois_metrics_data = {}
        for roi, metrics in data.items():
            for metric, samples in metrics.items():
                roi_metric_data = {"samples": [], "ages": [], "values": []}
                for sample in samples.keys():
                    roi_metric_data["samples"].append(sample)
                    roi_metric_data["ages"].append(meta[sample]["age"])
                    roi_metric_data["values"].append(samples[sample])

                rois_metrics_data.setdefault(roi, {})[metric] = roi_metric_data

        with open(
            "C:\\Users\\jerem\\Documents\\multiqc\\MultiQC_neuroimaging\\plot_ready_data.json", "w"
        ) as plot_ready_file:
            json.dump(dict(rois_metrics_data), plot_ready_file, indent=4)

        return dict(meta), dict(data), dict(rois_metrics_data)
