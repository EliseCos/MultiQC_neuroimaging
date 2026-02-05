import os
import json
from collections import defaultdict

import plotly.graph_objects as go
import plotly.io as pio

from neuroimaging.modules.harmonization.sections.section import SectionBundleMetric, HtmlContent

class BattacharyyaSection(SectionBundleMetric):
    HARMONIZATION_DISTANCE_SP_KEY  = "harmonization/harmonization_distance"

    def __init__(self, files):
        super().__init__()

        self.harmonization_distance_files = files

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
        self.data = {"raw": defaultdict(lambda: defaultdict(list)), "harmonized": defaultdict(lambda: defaultdict(list))}
        self.metrics = set()
        self.bundles = set()
        for f in self.harmonization_distance_files:
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
            
            self.bundles.update(set(bundles))

            metric = parts[1]
            self.metrics.add(metric)
            status = "harmonized" if "harmonized" in parts else "raw"

            for b, d in zip(bundles, distances):
                self.data[status][metric]["bundles"].append(b)
                self.data[status][metric]["distances"].append(float(d))

        # Create the boxplot using Plotly
        self.metrics = self.data["raw"].keys() | self.data["harmonized"].keys()

    
    @property
    def name(self):
        return "Mean Bhattacharyya distance (BD)"
    
    @property
    def anchor(self):
        return "harmonization_bhattacharyya"
    
    @property
    def description(self):
        return """
            The second subsection also includes a boxplot of the mean Bhattacharyya distance across
            bundles, before and after harmonization. The Bhattacharyya distance is a measure of similarity
            between two probability distributions. A lower Bhattacharyya distance indicates a higher
            similarity between the distributions. Thus, after harmonization, we expect to see a decrease
            in the Bhattacharyya distance values, indicating that the distributions of the moving site
            have become more similar to those of the reference site."""

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

    def build_html(self, default_metric: str, render_plot_func: str) -> HtmlContent:
        """
        Build the Bhattacharyya distance boxplots HTML content.

        Parameters:
        - default_metric: The metric to display by default when the page loads.
        - render_plot_func: The name of the JavaScript function to call to render the plots after switching the bundle or metric.
                            This function takes the parent div where the plots are located as an argument and renders the plots in that div.
        """
        bhattacharyya_html_content = ""
        b_div_ids = {}
        for metric in self.metrics:
            fig = go.Figure()

            # Raw data
            if metric in self.data["raw"]:
                fig.add_trace(
                    go.Box(
                        y=self.data["raw"][metric]["distances"],
                        name="Pre-Harmonization",
                        boxmean=True,
                        marker_color="darkblue",
                        text=self.data["raw"][metric]["bundles"],
                        boxpoints="all"
                    )
                )

            # Harmonized data
            if metric in self.data["harmonized"]:
                fig.add_trace(
                    go.Box(
                        y=self.data["harmonized"][metric]["distances"],
                        name="Post-Harmonization",
                        marker_color="#FF7C00",
                        text=self.data["harmonized"][metric]["bundles"],
                        boxmean=True,
                        boxpoints="all"
                    )
                )

            fig.update_layout(
                title=f"Mean Bhattacharyya distance across bundles<br>Metric: {metric}",
                title_xanchor="center",
                title_x=0.5,
                yaxis_title="Bhattacharyya Distance",
                height=500,
                legend_visible=False
            )

            b_div_id = f"bhattacharyya_{metric.replace(' ', '-')}"
            b_div_ids[metric] = b_div_id

            bhattacharyya_html_content += f"""
            <div id="{b_div_id}" style="max-width:800px; margin: 0 auto; display: {"none" if metric != default_metric else "block"};">
                {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
            </div>
            """
        
        render_bhatt_func = "renderBhattPlots"

        bhattacharyya_html_script = f"""
        <script>
        var bhattacharyya_div_ids = {json.dumps(b_div_ids)};
        var {render_bhatt_func} = function(metric) {{
            // Make every plot div hidden before showing the selected one
            Object.values(bhattacharyya_div_ids).forEach(function(divId) {{
                document.getElementById(divId).style.display = 'none';
            }});

            // Find ALL plotly graph divs inside it and resize each
            var divId = bhattacharyya_div_ids[metric];
            document.getElementById(divId).style.display = 'block';
            {render_plot_func}(document.getElementById(divId));
        }}
        </script>
        """

        data = HtmlContent(
            content=bhattacharyya_html_content + bhattacharyya_html_script,
            metadata={
                "render_metric_hook": render_bhatt_func
            }
        )
        return data