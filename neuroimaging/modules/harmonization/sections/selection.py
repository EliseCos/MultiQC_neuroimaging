from neuroimaging.modules.harmonization.sections.section import Section, HtmlContent
import logging

log = logging.getLogger("multiqc")


class SelectionSection(Section):
    """
    This section contains the button and the main javascript code
    to select the bundles and metrics to display in the harmonization
    results section.
    """

    def __init__(self):
        super().__init__()

        self.bundles = set()
        self.metrics = set()
        self.render_plot_func = "renderPlotlyPlots"
        self.on_bundle_metric_change_hooks = []
        self.on_metric_change_hooks = []

        self.default_metric = None
        self.default_bundle = None

    def set_bundles_metrics(self, bundles: set, metrics: set):
        """Set the available bundles and metrics for selection."""
        self.bundles = bundles
        self.metrics = metrics
        self.default_metric = list(self.metrics)[0] if len(self.metrics) > 0 else None
        self.default_bundle = list(self.bundles)[0] if len(self.bundles) > 0 else None

        log.info(f"SelectionSection: Available bundles: {self.bundles}")
        log.info(f"SelectionSection: Available metrics: {self.metrics}")

    def register_hook_on_bundle_metric_change(self, js_function_name: str):
        """Register a hook function to be called when the bundle or metric changes."""
        self.on_bundle_metric_change_hooks.append(js_function_name)

    def register_hook_on_metric_change(self, js_function_name: str):
        """Register a hook function to be called when the metric changes."""
        self.on_metric_change_hooks.append(js_function_name)

    def build_html(self) -> HtmlContent:
        # Sanity checks
        if self.default_metric is None:
            assert len(self.metrics) == 0, "Default metric is not set but there are available metrics."
        if self.default_bundle is None:
            assert len(self.bundles) == 0, "Default bundle is not set but there are available bundles."
        if (self.default_metric is None) and (self.default_bundle is None):
            raise ValueError("Metrics and bundles must be set before building the selection HTML content.")
        if len(self.metrics) == 0 and len(self.bundles) == 0:
            raise ValueError("At least one metric or one bundle must be available to build the selection HTML content.")

        html_bundle_dropdown = ""
        if len(self.bundles) > 0:
            html_bundle_dropdown = self._build_dropdown(
                label="Bundle",
                options=self.bundles,
                onchange_call="bundleMetricChanged(this.value, current_metric)",
                default_option=self.default_bundle,
            )

        html_metric_dropdown = ""
        if len(self.metrics) > 0:
            html_metric_dropdown = self._build_dropdown(
                label="Metric",
                options=self.metrics,
                onchange_call="bundleMetricChanged(current_bundle, this.value)",
                default_option=self.default_metric,
            )

        html_script = f"""
        <div class="d-flex justify-content-center gap-4">
            {html_bundle_dropdown}
            {html_metric_dropdown}
        </div>
        <script>
        var current_bundle = "{self.default_bundle}";
        var current_metric = "{self.default_metric}";

        // Utility function to re-render the plotly plots (useful to make sure they are correctly sized
        // and rendered when showing them).
        function {self.render_plot_func}(parentDiv) {{
            // Find ALL plotly graph divs inside it and resize each
            setTimeout(() => {{
                parentDiv.querySelectorAll('.plotly-graph-div').forEach(plotDiv => {{
                    Plotly.relayout(plotDiv, {{}});
                }});
            }}, 0);  // Even 0ms helps by deferring to next event loop
        }}

        function bundleMetricChanged(bundle, metric) {{
            var metric_changed = (metric !== current_metric);
            current_bundle = bundle;
            current_metric = metric;

            // Call any registered hooks on bundle or metric change (e.g. like the functions to change
            // the displayed harmonization plots when the bundle or metric changes)
            {";".join([f"if (typeof {hook} === 'function') {hook}(bundle, metric)" for hook in self.on_bundle_metric_change_hooks])};

            if (metric_changed) {{
                // Call any registered hooks on metric change (e.g. like the
                // functions to reload the Bhattacharyya plots when the metric changes)
                {";".join([f"if (typeof {hook} === 'function') {hook}(metric)" for hook in self.on_metric_change_hooks])};
            }}
        }}
        </script>
        """

        html_content = HtmlContent(content=html_script, metadata={})
        return html_content

    def _build_dropdown(self, label: str, options: set, onchange_call: str, default_option: str) -> str:
        """Helper function to build a dropdown HTML element."""
        html_options = "".join(
            [
                f'<option value="{option}" {"selected" if option == default_option else ""}>{option}</option>'
                for option in options
            ]
        )
        html_dropdown = f"""
        <div class="text-center">
            <label class="form-label mb-2 fw-semibold">{label}</label>
            <select class="form-select shadow-sm" 
                    aria-label="{label} selection" 
                    onchange="{onchange_call}"
                    style="min-width: 150px;">
                {html_options}
            </select>
        </div>
        """
        return html_dropdown
