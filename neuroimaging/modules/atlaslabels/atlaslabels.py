"""
=============================
MultiQC Atlas Labels Module
=============================

Project a volumetric atlas to cortical surfaces and visualize it with yabplot.
"""

import base64
from contextlib import redirect_stdout
import io
import logging
import os
import re
import tempfile
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import yabplot as yab

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """Project volumetric atlas labels into cortical/subcortical mesh previews."""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Atlas Labels",
            anchor="atlaslabels",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=(
                "This section contains QC images for the segmentation of cortical and subcortical structures, "
                "displayed as an overlay of anatomical labels. These labels are derived from structural MRI and "
                "serve as key regions of interest for connectivity analyses and volumetric measurements. To "
                "assess segmentation accuracy, verify that each label correctly corresponds to its respective "
                "anatomical structure. Too large or too small regions might indicate issues with the segmentation. "
                "A good first step is to investigate the volume reported in the global MultiQC report. "
                "If discrepancies are found, consider refining registration or segmentation parameters. "
                "It is worth noting that the 3D renderings are generated from the volumetric atlas "
                "for QC purposes only and may not perfectly capture the cortical surface details."
            ),
        )

        if not config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Fetch our input files and optional LUT
        self.config = getattr(config, "atlaslabels", {})
        nii_files = list(self.find_log_files("atlaslabels/volume"))
        lut_files = list(self.find_log_files("atlaslabels/metadata"))
        if not nii_files:
            raise ModuleNoSamplesFound

        # Get the indexes of the cortical and subcortical ROIs from user-defined config.
        cortical_idx_spec = self.config.get("cortical_rois_indexes")
        subcortical_idx_spec = self.config.get("subcortical_rois_indexes")
        if cortical_idx_spec is None and subcortical_idx_spec is None:
            log.warning(
                "Both cortical_rois_indexes and subcortical_rois_indexes must be provided in atlaslabels config."
            )
            raise ModuleNoSamplesFound

        # Parse the index specification.
        # If only one of cortical/subcortical specs are there, we assume the remaining
        # IDs are for the other group.
        cortical_ids = self._parse_index_spec(cortical_idx_spec)
        subcortical_ids = self._parse_index_spec(subcortical_idx_spec)
        if not cortical_ids and not subcortical_ids:
            log.warning("Cortical/subcortical ROI index specs are empty after parsing. Skipping Atlas Labels module.")
            raise ModuleNoSamplesFound

        # Ensure there is no overlap in case where user defined both indexes.
        overlap_ids = cortical_ids.intersection(subcortical_ids)
        if overlap_ids:
            log.warning(
                "Atlas labels index selections overlap across cortical/subcortical groups. "
                "Removing %d overlapping IDs from subcortical group.",
                len(overlap_ids),
            )
            subcortical_ids = subcortical_ids.difference(overlap_ids)
            # Make sure we still have some subcortical IDs left after removing the overlap.
            if not subcortical_ids:
                log.warning("No subcortical ROI IDs remain after overlap filtering. Skipping Atlas Labels module.")
                raise ModuleNoSamplesFound

        # Use the LUT files if available
        lut_file = None
        if not lut_files:
            log.warning("No atlas metadata file found. A default colormap will be used.")
            all_regions = {}
        else:
            lut_file = lut_files[0]
            all_regions = self._parse_metadata_lines(lut_file.get("f", ""))

        nii_file = nii_files[0]
        nii_path = self._resolve_found_file_path(nii_file)

        # Extract the ROIs based on the indexes and fill in the colormap from the LUT if available
        cortical_regions = self._build_regions_from_ids(cortical_ids, all_regions, name_prefix="Cortical")
        subcortical_regions = self._build_regions_from_ids(subcortical_ids, all_regions, name_prefix="Subcortical")

        # Small check to make sure we have valid regions.
        if len(cortical_regions) == 0 and len(subcortical_regions) == 0:
            log.warning("No valid regions remain after parsing/filtering. Skipping Atlas Labels module.")
            raise ModuleNoSamplesFound

        # Setting temp directories.
        work_dir = Path(tempfile.mkdtemp(prefix="multiqc_atlaslabels_"))
        cortical_dir = Path(self.config.get("cortical_out_dir", str(work_dir / "cortical_atlas")))
        subcortical_dir = Path(self.config.get("subcortical_out_dir", str(work_dir / "subcortical_atlas")))
        cortical_preview_path = work_dir / "cortical_preview.png"
        sub_preview_path = work_dir / "subcortical_preview.png"

        try:
            # Set up the cmap for the cortical regions.
            cortical_plot_regions = {**cortical_regions}
            cortical_labels = {rid: info["name"] for rid, info in sorted(cortical_plot_regions.items())}
            cortical_data, cortical_cmap, cortical_vminmax = self._build_discrete_mapping(
                cortical_plot_regions,
                fallback_cmap_name=self.config.get("cortical_cmap", "viridis"),
                force_cmap=self.config.get("force_cortical_cmap", False),
            )

            # Redirect log output from yabplot to avoid cluttering the MultiQC report logs with non-critical messages.
            f = io.StringIO()
            with redirect_stdout(f):
                yab.build_subcortical_atlas(
                    nii_path=nii_path,
                    labels_dict=cortical_labels,
                    out_dir=str(cortical_dir),
                    smooth_i=self.config.get("smooth_i", 20),
                    smooth_f=self.config.get("smooth_f", 0.7),
                )
            log.debug(f"yabplot cortical atlas build output:\n{f.getvalue()}")

            # Generates the images, we use the subcortical function since it uses
            # surfaces generated from a volumetric parcellation
            plotter_cort = yab.plot_subcortical(
                data=cortical_data,
                custom_atlas_path=str(cortical_dir),
                bmesh_type=None,
                views=self.config.get(
                    "views",
                    ["left_lateral", "left_medial", "superior", "anterior"],
                ),
                cmap=cortical_cmap,
                vminmax=cortical_vminmax,
                style=self.config.get("style", "glossy"),
                figsize=tuple(self.config.get("figsize", [1200, 450])),
                display_type="object",
                export_path=str(cortical_preview_path),
            )
            if hasattr(plotter_cort, "close"):
                plotter_cort.close()

            # Embed the image as base64 in the report
            if cortical_preview_path.exists():
                img_b64 = base64.b64encode(cortical_preview_path.read_bytes()).decode("ascii")
                cortical_content = (
                    '<img alt="Cortical atlas mesh preview" '
                    'style="max-width:auto;height:auto;" '
                    f'src="data:image/png;base64,{img_b64}" />'
                )
        except Exception as e:
            log.warning(f"Cortical atlas build/render failed: {e}")
            cortical_content = "<p>Failed to generate cortical atlas preview.</p>"

        try:
            sub_labels = {rid: info["name"] for rid, info in sorted(subcortical_regions.items())}
            # Redirect log output from yabplot to avoid cluttering the MultiQC report logs with non-critical messages.
            f = io.StringIO()
            with redirect_stdout(f):
                yab.build_subcortical_atlas(
                    nii_path=nii_path,
                    labels_dict=sub_labels,
                    out_dir=str(subcortical_dir),
                    smooth_i=self.config.get("smooth_i", 20),
                    smooth_f=self.config.get("smooth_f", 0.7),
                )
            log.debug(f"yabplot subcortical atlas build output:\n{f.getvalue()}")

            # Similar to the cortical region, build the colormap either based on the LUT
            # or the default colormap.
            sub_plot_regions = dict(subcortical_regions)
            sub_data, sub_cmap, sub_vminmax = self._build_discrete_mapping(
                sub_plot_regions,
                fallback_cmap_name=self.config.get("subcortical_cmap", self.config.get("cmap", "viridis")),
                force_cmap=self.config.get("force_subcortical_cmap", False),
            )

            plotter_sub = yab.plot_subcortical(
                data=sub_data,
                custom_atlas_path=str(subcortical_dir),
                bmesh_type=None,
                views=self.config.get(
                    "views",
                    ["left_lateral", "left_medial", "superior", "anterior"],
                ),
                cmap=sub_cmap,
                vminmax=sub_vminmax,
                style=self.config.get("style", "glossy"),
                figsize=tuple(self.config.get("figsize", [1200, 450])),
                display_type="object",
                export_path=str(sub_preview_path),
            )
            if hasattr(plotter_sub, "close"):
                plotter_sub.close()

            # Embed the image as base64 in the report
            if sub_preview_path.exists():
                sub_img_b64 = base64.b64encode(sub_preview_path.read_bytes()).decode("ascii")
                subcortical_content = (
                    '<img alt="Subcortical atlas mesh preview" '
                    'style="max-width:auto;height:auto;" '
                    f'src="data:image/png;base64,{sub_img_b64}" />'
                )
        except Exception as e:
            log.warning(f"Subcortical atlas build/render failed: {e}")
            subcortical_content = "<p>Failed to generate subcortical atlas preview.</p>"

        # Add the sections ot the report using base MultiQC functions
        self.add_section(
            name="Cortical parcellation",
            anchor="atlaslabels-cortical-preview",
            content=cortical_content,
        )

        self.add_section(
            name="Subcortical parcellation",
            anchor="atlaslabels-subcortical-preview",
            content=subcortical_content,
        )

        self.write_data_file(
            {
                "atlas_nifti": nii_path,
                "atlas_metadata_source": lut_file.get("fn", "") if lut_file else "",
                "cortical_output_dir": str(cortical_dir),
                "cortical_region_count": len(cortical_regions),
                "cortical_index_spec": cortical_idx_spec,
                "subcortical_region_count": len(subcortical_regions),
                "subcortical_output_dir": str(subcortical_dir),
                "subcortical_index_spec": subcortical_idx_spec,
            },
            "multiqc_atlaslabels",
        )

    @staticmethod
    def _resolve_found_file_path(found_file: dict) -> str:
        fn = found_file.get("fn", "")
        if os.path.isabs(fn):
            return fn
        root = found_file.get("root", "")
        return os.path.join(root, fn) if root else fn

    @staticmethod
    def _build_discrete_mapping(
        regions: dict[int, dict], fallback_cmap_name: str = "viridis", force_cmap: bool = False
    ) -> tuple[dict[str, float], ListedColormap | object, list[float], str]:
        """Build deterministic region->value map and colors.

        Uses LUT RGB values only when every region has RGB fields.
        Otherwise, falls back to a matplotlib colormap.
        """
        values: dict[str, float] = {}
        colors = []
        has_complete_rgb = bool(regions) and all(all(k in info for k in ("r", "g", "b")) for info in regions.values())

        for idx, (rid, info) in enumerate(sorted(regions.items()), start=1):
            values[info["name"]] = float(idx)
            if has_complete_rgb and not force_cmap:
                r, g, b = int(info["r"]), int(info["g"]), int(info["b"])
                colors.append((r / 255.0, g / 255.0, b / 255.0))

        if not colors:
            cmap = colormaps.get_cmap(fallback_cmap_name)
            vminmax = [1, max(1, len(values))]
            return values, cmap, vminmax

        cmap = ListedColormap(colors)
        vminmax = [1, len(colors)]
        return values, cmap, vminmax

    @staticmethod
    def _parse_index_spec(index_spec) -> set[int]:
        """Parse flexible ROI index specification from config.

        Supported examples:
          - 12
          - "12"
          - "1:188" (inclusive)
          - "1-188" (inclusive)
          - [1, 2, "5:10", "20-30"]
          - [{"1:188": null}] (defensive support for odd YAML flow parsing)
        """
        parsed: set[int] = set()

        def _parse_one(item):
            if item is None:
                return
            if isinstance(item, int):
                parsed.add(item)
                return
            if isinstance(item, str):
                token = item.strip()
                if not token:
                    return
                range_match = re.match(r"^(-?\d+)\s*[:-]\s*(-?\d+)$", token)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    step = 1 if end >= start else -1
                    parsed.update(range(start, end + step, step))
                    return
                if re.match(r"^-?\d+$", token):
                    parsed.add(int(token))
                    return
                for part in token.split(","):
                    if part.strip() != token:
                        _parse_one(part)
                return
            if isinstance(item, dict):
                for k, v in item.items():
                    _parse_one(k)
                    _parse_one(v)
                return
            if isinstance(item, (list, tuple, set)):
                for sub in item:
                    _parse_one(sub)

        _parse_one(index_spec)
        return parsed

    @staticmethod
    def _build_regions_from_ids(
        selected_ids: set[int], all_regions: dict[int, dict], name_prefix: str
    ) -> dict[int, dict]:
        """Build a region dict from selected IDs, falling back to synthetic names when LUT is missing."""
        regions: dict[int, dict] = {}
        for rid in sorted(selected_ids):
            if rid in all_regions:
                regions[rid] = dict(all_regions[rid])
            else:
                regions[rid] = {"name": f"{name_prefix}_{rid}"}
        return regions

    @staticmethod
    def _parse_metadata_lines(text: str) -> dict[int, dict]:
        """Parse atlas metadata lines into {id: {name, r, g, b}}.

        Supported formats (delimiter: whitespace, comma, or semicolon):
          - BIDS dseg / ITK-SNAP: ``id  name  R  G  B  A``
          - RGB only:             ``id  name  R  G  B``
          - Name only:            ``id  name``
          - Names with spaces:    ``id  Left Region  R  G  B  A``

        Background (id 0) is always skipped.
        """
        regions: dict[int, dict] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [t for t in re.split(r"[\s,;]+", line) if t]
            if len(tokens) < 2:
                continue
            try:
                region_id = int(tokens[0])
            except ValueError:
                continue
            if region_id == 0:
                continue  # skip background
            # Try to detect a trailing RGBA (4 tokens) or RGB (3 tokens) block.
            color: dict = {}
            name_end = len(tokens)
            for n_color in (4, 3):
                if len(tokens) >= 2 + n_color:
                    try:
                        vals = [int(t) for t in tokens[-n_color:]]
                        color = {"r": vals[0], "g": vals[1], "b": vals[2]}
                        if n_color == 4:
                            color["a"] = vals[3]
                        name_end = len(tokens) - n_color
                        break
                    except ValueError:
                        pass
            region_name = " ".join(tokens[1:name_end]).strip("\",'")
            if region_name:
                regions[region_id] = {"name": region_name, **color}
        return regions
