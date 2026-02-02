import os
import json
import logging
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuroimaging.modules.harmonization.sections.section import Section, HtmlContent
from neuroimaging.modules.harmonization.utils import load_sites_data, PALETTE, to_plotly_rgb
from neuroimaging.modules.harmonization.plotjson import PlotJsonAggregator

log = logging.getLogger("multiqc")

class DistributionSection(Section):
    REF_STATS_SP_KEY            = "harmonization/reference_stats"
    RAW_STATS_SP_KEY            = "harmonization/raw_stats"
    HARMONIZED_STATS_SP_KEY     = "harmonization/harmonized_stats"
    DATA_MODELS_PLOTS_SP_KEY    = "harmonization/data_models_plots"
    HARMONIZATION_PLOTS_SP_KEY  = "harmonization/harmonization_plots"

    def __init__(
        self,
        ref_stats_file,
        raw_stats_files,
        harmonized_stats_files,
        data_model_plots_files,
        harmonization_plots_files
    ):
        super().__init__()
        self.ref_df        = load_sites_data(ref_stats_file)
        self.raw_df        = load_sites_data(raw_stats_files)
        self.harmonized_df = load_sites_data(harmonized_stats_files)
        
        # Read plot data from JSON files and aggregate
        self.data_model_json    = PlotJsonAggregator.from_json(list(map(lambda f: f["f"], data_model_plots_files)))
        self.harmonization_json = PlotJsonAggregator.from_json(list(map(lambda f: f["f"], harmonization_plots_files)))

        # Extract list of metrics and non-metric columns
        headers = set(self.harmonized_df.columns)
        meta_columns = set(["site", "age", "sex", "handedness", "disease"])
        non_metric_columns = set(["sample", "roi"]) | meta_columns
        
        self.metrics = headers - non_metric_columns
        self.bundles = list(self.harmonized_df["roi"].unique())

        # Verify that the raw_df and harmonized_df have the same samples
        assert set(self.raw_df["sample"]) == set(self.harmonized_df["sample"]), \
            "The raw and harmonized data files must contain the same samples."
        
        # Verify that the raw and harmonized data have the same ROIs
        assert set(self.raw_df["roi"]) == set(self.harmonized_df["roi"]), \
            "The raw and harmonized data files must contain the same ROIs."
        
        # Verify that the dataframes and the jsons have the same ROIs and metrics
        for bundle in self.bundles:
            for metric in self.metrics:
                assert bundle in self.data_model_json.data, \
                    f"The bundle {bundle} is missing from the data model JSON."
                assert metric in self.data_model_json.data[bundle], \
                    f"The metric {metric} is missing from the data model JSON for bundle {bundle}."
                assert bundle in self.harmonization_json.data, \
                    f"The bundle {bundle} is missing from the harmonization JSON."
                assert metric in self.harmonization_json.data[bundle], \
                    f"The metric {metric} is missing from the harmonization JSON for bundle {bundle}."
    
    @property
    def name(self):
        return "Distributional results"
    
    @property
    def anchor(self):
        return "harmonization_distributions"
    
    @property
    def description(self):
        return """
            The graphs include data model plots, which we use to visualize
            a scatter plot and a regression line of the reference and
            moving sites, before harmonization. We also include pre and
            post-harmonization age curves to visualize how the data
            distributions change after harmonization. The user should make
            sure that the post-harmonization age curves of the moving site
            better align with the reference age curves."""

    def get_metrics(self):
        """Get the list of metrics available for plotting."""
        return sorted(list(self.metrics))
    
    def filter_metrics(self, metrics_to_keep):
        """Filter the metrics to keep only those specified."""
        self.metrics = set(metrics_to_keep) & self.metrics
    
    def get_bundles(self):
        """Get the list of bundles available for plotting."""
        return sorted(self.bundles)
    
    def filter_bundles(self, bundles_to_keep):
        """Filter the bundles to keep only those specified."""
        self.bundles = list(set(bundles_to_keep) & set(self.bundles))

    def build_html(self) -> HtmlContent:
        is_first_tuple = True
        all_div_ids = []
        anchors = {}
        plots_html_content = ""
        for bundle in self.bundles:
            for metric in self.metrics:
                log.info(f"Harmonization QC: Processing bundle: {bundle}, metric: {metric}")
                html, div_id = self.make_subplot_row(bundle, metric, hidden=not is_first_tuple)

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
                for i, bundle in enumerate(self.bundles)
            ]
        )
        html_metric_options = "".join(
            [
                f'<option value="{metric}" {"selected" if i == 0 else ""}>{metric}</option>'
                for i, metric in enumerate(self.metrics)
            ]
        )
        default_bundle = list(self.bundles)[0]
        default_metric = list(self.metrics)[0]
        render_bhatt_func = "renderBhattacharyyaPlots"
        render_plot_func = "renderPlotlyPlots"

        html_script = f"""
        <div class="d-flex justify-content-center gap-4">
            <div class="text-center">
                <label class="form-label mb-2 fw-semibold">Bundle</label>
                <select class="form-select shadow-sm" 
                        aria-label="Bundle selection" 
                        onchange="showBundleMetricPlot(this.value, current_metric)"
                        style="min-width: 150px;">
                    {html_bundle_options}
                </select>
            </div>

            <div class="text-center">
                <label class="form-label mb-2 fw-semibold">Metric</label>
                <select class="form-select shadow-sm" 
                        aria-label="Metric selection" 
                        onchange="showBundleMetricPlot(current_bundle, this.value)"
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

        function {render_plot_func}(parentDiv) {{
            // Find ALL plotly graph divs inside it and resize each
            setTimeout(() => {{
                parentDiv.querySelectorAll('.plotly-graph-div').forEach(plotDiv => {{
                    Plotly.relayout(plotDiv, {{}});
                }});
            }}, 0);  // Even 0ms helps by deferring to next event loop
        }}

        // Placeholder as this is should be implemented in bhattacharyya section
        var {render_bhatt_func};

        function showBundleMetricPlot(bundle, metric) {{
            var metric_changed = (metric !== current_metric);
            current_bundle = bundle;
            current_metric = metric;

            // Make sure all plot divs are hidden before showing the selected one
            div_ids.forEach(function(divId) {{
                document.getElementById(divId).style.display = 'none';
            }});
            
            // Show the selected scenario
            var selected_div_id = anchors[bundle][metric];
            document.getElementById(selected_div_id).style.display = 'block';

            const subPlotsParentDiv = document.getElementById(selected_div_id);
            {render_plot_func}(subPlotsParentDiv);

            // Also render the bhattacharyya_plot_div plot if the metric is changed
            // and the function is defined.
            if (metric_changed && {render_bhatt_func}) {{
                {render_bhatt_func}(metric);
            }}
        }}
        </script>
        """

        data = HtmlContent(
            content=html_script + plots_html_content,
            metadata={
                "default_bundle": default_bundle,
                "default_metric": default_metric,
                "render_bhatt_func": render_bhatt_func,
                "render_plot_func": render_plot_func

            }
        )

        return data

    def make_subplot_row(self, bundle, metric, hidden=False):
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
        ref_median_curve = self._plot_percentiles_and_get_median_curve(
            reference_percentile_plots, fig, row, col, is_moving=False)

        ############################
        # Moving percentiles (raw)
        ############################
        moving_percentile_plots = plot_json['moving_raw_percentiles']
        mov_median_curve = self._plot_percentiles_and_get_median_curve(
            moving_percentile_plots, fig, row, col, is_moving=True)


        ############################
        # Median curves
        ############################
        # Plot the median curves last so they get drawn on top of the filled areas.
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
        ref_median_curve = self._plot_percentiles_and_get_median_curve(reference_percentile_plots, fig, row, col, is_moving=False)

        ############################
        # Moving percentiles (harmonized)
        ############################
        moving_percentile_plots = plot_json['moving_harmonized_percentiles']
        mov_median_curve = self._plot_percentiles_and_get_median_curve(moving_percentile_plots, fig, row, col, is_moving=True)

        ############################
        # Median curves
        ############################
        # Plot the median curves last so they get drawn on top of the filled areas.
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

    def _plot_percentiles_and_get_median_curve(self, data: dict, fig: go.Figure, row: int, col: int, is_moving: bool) -> dict:
        """
        This is a helper function to plot percentile filled areas on the given figure.
        We expect an odd number of percentiles so that there is a middle percentile (50)
        representing the median. This function fills the area between matching percentiles
        (e.g., 5th and 95th, 10th and 90th, etc.) and returns the key data to plot the median curve.
        We do not plot the median curve here to ensure it is drawn on top of the filled areas.
        
        :param data: Input dictionnary containing percentile curves information. We expect
            each key to be a percentile plot name (we expect an odd number of percentiles,
            which includes a percentile 50 for the median) and each value to be a dict with
            keys: 'percentile', 'data_x', 'data_y' and 'color'.
             - 'percentile': int: The percentile number (e.g., 5, 10, 25, 50, 75, 90, 95).
             - 'data_x': list of float: The x values (ages).
             - 'data_y': list of float: The y values (metric values).
             - 'color': str: The color to use for the percentile curve fill and line.
        :type data: dict
        :param fig: Figure onto which to plot the percentiles.
        :type fig: go.Figure
        :param row: Row number in the subplot figure.
        :type row: int
        :param col: Column number in the subplot figure.
        :type col: int
        :param is_moving: Whether the percentiles are for the moving site (True) or reference site (False).
        :type is_moving: bool
        """
        # Extract the percentile curves dict keys ordered by percentile number.
        percentile_plots = [k for k in data.keys()]
        percentiles = [v["percentile"] for v in data.values()]
        available_percentiles = [p for _, p in sorted(zip(percentiles, percentile_plots))]

        # Make sure the number of percentiles is odd to have a middle one
        if len(available_percentiles) % 2 == 0:
            raise ValueError("The number of percentiles must be odd to have a main/mean percentile.")

        middle = len(available_percentiles) // 2
        other_percentiles = available_percentiles[:middle] + available_percentiles[middle+1:]

        # Make sure the 5th percentile is matched with the 95th, 10th with 90th, etc.
        for i in range(len(other_percentiles)//2):
            p1_name = other_percentiles[i]
            p2_name = other_percentiles[-(i+1)]

            p1 = data[p1_name]
            p2 = data[p2_name]
            
            # We need to fill the area between p1 and p2
            fig.add_trace(
                go.Scatter(
                    x=p1['data_x'] + p2['data_x'][::-1],
                    y=p1['data_y'] + p2['data_y'][::-1],
                    fill='toself',
                    fillcolor=p1["color"],
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{'Moving' if is_moving else 'Reference'} {p1['percentile']}th - {p2['percentile']}th percentile",
                    legendgroup=f"{'moving_group' if is_moving else 'reference_group'}",
                    showlegend=False,
                    opacity=0.2
                ),
                row=row,
                col=col
            )
        
        median_key = available_percentiles[middle]

        return median_key
