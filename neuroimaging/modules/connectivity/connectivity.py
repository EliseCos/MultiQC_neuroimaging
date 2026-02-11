"""
===========================================
Connectivity Module
===========================================

This module collects and visualizes connectivity matrices
fromm connectome analysis. It supports the following file types:
    * NumPy (.npy) connectivity matrices (for now)

For global reports, it computes the density of the connectivity
matrice, and plots the distribution across subjects in a violin plot.
It also generates a heatmap of the average connectivity matrix
across all subjects.

For subject reports, it visualizes the individual connectivity
matrix as a heatmap.
"""

import logging
import re
from typing import Dict

import numpy as np

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import heatmap, violin

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """ "MultiQC module for connectivity matrices."""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Structural Connectivity",
            anchor="connectivity",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Assessment of structural connectivity for quality control."
            " For each subject, the density of the connectivity matrix is computed,"
            " and subjects with unusually low density are flagged. Additionally,"
            " the average connectivity matrix across all subjects is visualized.",
        )

        #  Get the single-subject mode if set.
        single_subject_mode = config.get("single_subject", False)

        # Find and parse connectivity matrix files
        conn_data = {}
        config_fp = config.sp.get("connectivity", {}).get("fn", "")
        for f in self.find_log_files("connectivity"):
            parsed = self.parse_connectivity_file(f, config_fp)
            if parsed:
                sample_name = parsed["sample_name"]
                conn_data[sample_name] = parsed["values"]

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names.
        conn_data = self.filter_samples(conn_data)

        if len(conn_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(conn_data)} samples.")

        # Generate global plots if not in single-subject mode
        if not single_subject_mode:
            # Compute densities
            densities = {s_name: {"density": self.compute_density(matrix)} for s_name, matrix in conn_data.items()}

            # Add to general stats table.
            self.general_stats_addcols(
                densities,
                {
                    "density": {
                        "title": "Structural Connectivity Density",
                        "description": "Density of the connectivity matrix (proportion of non-zero connections).",
                        "min": min(d["density"] for d in densities.values()),
                        "max": max(d["density"] for d in densities.values()),
                        "format": "{:.2f}",
                    }
                },
            )

    def parse_connectivity_file(self, f: str, config_fp: str) -> Dict:
        """Parse a connectivity matrix file.

        Args:
            f (str): Path to the connectivity matrix file.
            config_fp (str): Configured file path pattern to identify connectivity files.

        Returns:
            Dict: Parsed data including sample name and connectivity values.
        """

        values = np.load(f)

        # Extract and clean sample name from filename.
        # Using the pattern from custom_code.py for consistency.
        # Remove the pattern suffix from filename to get the sample name.
        filename = f["fn"]
        pattern_suffix = config_fp.lstrip("*")
        if pattern_suffix and filename.endswith(pattern_suffix):
            # Similar to other modules, remove the suffix and any trailing underscores or hyphens.
            sample_name = re.sub(r"_+$|-+$", "", filename[: -len(pattern_suffix)])
        else:
            # Fallback to the default way of cleaning sample names.
            sample_name = f["s_name"]

        # Apply MultiQC's sample name cleaning
        sample_name = self.clean_s_name(sample_name, f)

        return {"sample_name": sample_name, "values": values}

    def compute_density(self, matrix: np.ndarray) -> float:
        """Compute the density of a connectivity matrix.

        Args:
            matrix (np.ndarray): Connectivity matrix.

        Returns:
            float: Density of the connectivity matrix.
        """
        # Count non-zero connections
        non_zero_connections = np.count_nonzero(matrix)
        total_connections = matrix.size
        density = non_zero_connections / total_connections
        return density
