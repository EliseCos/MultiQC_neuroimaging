"""
=============================
Bundles modules
=============================
This module visualizes bundles to facilitate the
quality control process.

This module does not leverage any metrics and consist
solely of a visual quality control step. To leverage metrics
per bundles, please refer to the tractometry or metricsinroi
modules.
"""

import base64
import html
import io
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import redirect_stdout
from typing import Dict, List

import nibabel as nib
import numpy as np
import pyvista as pv
import yabplot as yab
from scipy.ndimage import gaussian_filter
from skimage import io as skio
from skimage import measure

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """
    This module visualizes bundles to facilitate the
    quality control process.

    This module does not leverage any metrics and consist
    solely of a visual quality control step. To leverage metrics
    per bundles, please refer to the tractometry or metricsinroi
    modules.
    """

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="White Matter Bundles",
            target="bundles",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=(
                "Visualization of white matter bundles for quality control purposes. "
                "Use the drop-down menu to switch between bundles and assess the quality "
                "of the extraction process. Specifically, you can check the overall shape "
                "and anatomical location, as well as the presence of spurious streamlines. "
            ),
        )

        if not config.kwargs.get("single_subject"):
            raise ModuleNoSamplesFound

        self.config = getattr(config, "bundles", {}) or {}

        trk_files = list(self.find_log_files("bundles/trk"))
        nii_files = list(self.find_log_files("bundles/nii"))

        if not trk_files:
            raise ModuleNoSamplesFound

        if len(nii_files) > 1:
            log.warning("Found multiple .nii files, will use the first one.")
        nii_file = self._resolve_found_file_path(nii_files[0]) if nii_files else None
        if not nii_file:
            log.warning("No .nii files found, bundle visualizations will be performed without background.")

        bundles_to_include = self.config.get("bundles", [])
        if bundles_to_include:
            trk_dict = {}
            for bundle_name in bundles_to_include:
                matching_files = [
                    found_file
                    for found_file in trk_files
                    # Look for the bundle name anywhere in the filename.
                    if re.search(
                        rf"(^|[_-]){re.escape(bundle_name)}([_.-]|$)",
                        found_file.get("fn", ""),
                        re.IGNORECASE,
                    )
                ]
                if matching_files:
                    trk_dict[bundle_name] = self._resolve_found_file_path(matching_files[0])
                else:
                    log.warning(f"No .trk file found for bundle '{bundle_name}'. Skipping this bundle.")
            if not trk_dict:
                raise ModuleNoSamplesFound(f"No .trk files found matching the specified bundles: {bundles_to_include}")
        else:
            trk_dict = self._parse_trk_paths(trk_files)
            if not trk_dict:
                log.warning(
                    "No valid .trk files found to parse bundle names. Please check the file naming convention "
                    "and ensure that .trk files are present."
                )
                raise ModuleNoSamplesFound("No valid .trk files found to parse bundle names.")

        log.info(f"Found {len(trk_dict)} bundles to visualize")
        bg_mesh = self._load_nii_as_mesh(nii_file) if nii_file else None

        bundle_images = self._render_bundle_images(trk_dict, bg_mesh)
        if not bundle_images:
            raise ModuleNoSamplesFound("No bundle previews could be generated.")

        default_bundle = next(iter(bundle_images))
        content = self._build_bundle_browser(bundle_images, default_bundle)

        self.add_section(
            name="Bundle Visualization",
            anchor="bundles_visualization",
            description=(
                "Use the dropdown to switch between bundle previews. The figure shows the bundle extraction "
                "for one tract at a time so you can inspect anatomy and spurious streamlines more easily. "
                "<b>Views include, from left to right, superior, right lateral, left lateral, and anterior.</b> "
                "In the left and right views, the view not containing the specific bundle will be empty "
                "(this might not work with the corpus callosum bundles). "
                "To rapidly switch between bundles, you can also use the left and right arrow keys on your keyboard. "
            ),
            content=content,
        )

    def _resolve_found_file_path(self, found_file: dict) -> str:
        fn = found_file.get("fn", "")
        if os.path.isabs(fn):
            return fn
        root = found_file.get("root", "")
        return os.path.join(root, fn) if root else fn

    def _parse_trk_paths(self, files: List[dict]) -> Dict[str, str]:
        """Parse bundle names from found .trk files and return a bundle->path mapping."""
        bundle_dict: Dict[str, str] = {}
        for found_file in files:
            filename = found_file.get("fn", "")
            match = re.search(r"_tract-([^_]+)", filename)
            if match:
                bundle_name = match.group(1)
                bundle_dict[bundle_name] = self._resolve_found_file_path(found_file)
            else:
                log.warning(f"Could not extract bundle name from file path: {filename}. Skipping this file.")
        return bundle_dict

    def _render_bundle_images(self, trk_dict: Dict[str, str], bg_mesh: pv.PolyData | None) -> Dict[str, str]:
        """Render bundle figures and return base64-encoded HTML images keyed by bundle name."""
        bundle_images: Dict[str, str] = {}

        for bundle_name, trk_path in trk_dict.items():
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_trk_path = os.path.join(tmpdir, f"{bundle_name}.trk")
                    shutil.copy(trk_path, tmp_trk_path)

                    export_path = os.path.join(tmpdir, f"bundles_{bundle_name}.png")
                    with redirect_stdout(io.StringIO()):
                        yab.plot_tracts(
                            custom_atlas_path=tmpdir,
                            views=["superior", "right_lateral", "left_lateral", "anterior"],
                            bmesh=bg_mesh,
                            figsize=(2000, 1200),
                            display_type="object",
                            orientation_coloring=True,
                            export_path=export_path,
                        )

                    if not os.path.exists(export_path):
                        log.warning(f"Bundle preview was not created for '{bundle_name}'.")
                        continue

                    self._crop_png_whitespace(export_path)

                    with open(export_path, "rb") as fp:
                        img_b64 = base64.b64encode(fp.read()).decode("ascii")

                    bundle_images[bundle_name] = (
                        '<img alt="{alt}" style="width: 100%; max-width: 100%; height: auto; display: block;" '
                        'src="data:image/png;base64,{data}" />'
                    ).format(alt=html.escape(f"Bundle preview for {bundle_name}"), data=img_b64)
            except Exception as exc:
                log.warning(f"Failed to render bundle preview for '{bundle_name}': {exc}")

        return bundle_images

    def _build_bundle_browser(self, bundle_images: Dict[str, str], default_bundle: str) -> str:
        """Build the dropdown and the one-at-a-time bundle preview container."""
        bundle_ids = {
            bundle_name: f"bundle_{index}_{re.sub(r'[^0-9a-zA-Z_]', '_', bundle_name)}"
            for index, bundle_name in enumerate(bundle_images)
        }

        selector_html = ""
        if len(bundle_images) > 1:
            options_html = "".join(
                [
                    f'<option value="{html.escape(bundle_name)}" '
                    f"{'selected' if bundle_name == default_bundle else ''}>"
                    f"{html.escape(bundle_name)}</option>"
                    for bundle_name in bundle_images
                ]
            )
            selector_html = f"""
            <div class="d-flex justify-content-center mb-3">
                <div class="text-center">
                    <label class="form-label mb-2 fw-semibold">Bundle</label>
                    <div class="d-flex align-items-center gap-2">
                        <button
                            class="btn btn-outline-secondary"
                            type="button"
                            aria-label="Previous bundle"
                            onclick="showPreviousBundle()"
                        >&larr;</button>
                        <select
                            id="bundle-selector"
                            class="form-select shadow-sm"
                            aria-label="Bundle selection"
                            onchange="renderBundleView(this.value)"
                            style="min-width: 220px;"
                        >
                            {options_html}
                        </select>
                        <button
                            class="btn btn-outline-secondary"
                            type="button"
                            aria-label="Next bundle"
                            onclick="showNextBundle()"
                        >&rarr;</button>
                    </div>
                </div>
            </div>
            """

        bundle_panels = []
        for bundle_name, img_html in bundle_images.items():
            display = "block" if bundle_name == default_bundle else "none"
            bundle_panels.append(
                f'<div id="{bundle_ids[bundle_name]}" style="display: {display}; width: 100%;">{img_html}</div>'
            )

        script = f"""
        <script>
        var bundleContainers = {json.dumps(bundle_ids)};
        var bundleOrder = {json.dumps(list(bundle_images.keys()))};
        var currentBundleIdx = bundleOrder.indexOf({json.dumps(default_bundle)});

        function updateBundleSelector(bundleName) {{
            var selector = document.getElementById('bundle-selector');
            if (selector) {{
                selector.value = bundleName;
            }}
        }}

        function renderBundleView(bundleName) {{
            Object.keys(bundleContainers).forEach(function(name) {{
                var container = document.getElementById(bundleContainers[name]);
                if (container) {{
                    container.style.display = "none";
                }}
            }});

            var selectedContainer = document.getElementById(bundleContainers[bundleName]);
            if (selectedContainer) {{
                selectedContainer.style.display = "block";
                currentBundleIdx = bundleOrder.indexOf(bundleName);
                updateBundleSelector(bundleName);
            }}
        }}

        function showPreviousBundle() {{
            if (!bundleOrder.length) {{
                return;
            }}
            currentBundleIdx = (currentBundleIdx - 1 + bundleOrder.length) % bundleOrder.length;
            renderBundleView(bundleOrder[currentBundleIdx]);
        }}

        function showNextBundle() {{
            if (!bundleOrder.length) {{
                return;
            }}
            currentBundleIdx = (currentBundleIdx + 1) % bundleOrder.length;
            renderBundleView(bundleOrder[currentBundleIdx]);
        }}

        if (!window.bundleArrowKeyListenerAttached) {{
            window.bundleArrowKeyListenerAttached = true;
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'ArrowLeft') {{
                    event.preventDefault();
                    showPreviousBundle();
                }} else if (event.key === 'ArrowRight') {{
                    event.preventDefault();
                    showNextBundle();
                }}
            }});
        }}
        </script>
        """

        return selector_html + "\n" + "\n".join(bundle_panels) + script

    @staticmethod
    def _load_nii_as_mesh(
        nii_fp,
        threshold: float = 0.5,
        blur_sigma: float = 1.5,
        smooth_i: int = 10,
        smooth_f: float = 0.1,
    ) -> pv.PolyData:
        """
        Small utility function to load a nifti image, then convert it to
        meshes using marching cubes.

        **For now, this function is here, but will be included directly in
        yabplot in the future.**

        Parameters
        ----------
        nii_path : str
            Absolute path to a NIfTI file representing a 3D volume. If 4D, only the first volume will be used.
        threshold : float, optional
            Threshold applied after optional blur. Voxels ``> threshold`` are kept.
        blur_sigma : float, optional
            Gaussian blur (voxel units) before thresholding.
        smooth_i : int, optional
            Number of PyVista smoothing iterations after surface extraction.
        smooth_f : float, optional
            Relaxation factor for mesh smoothing.

        Returns
        -------
        mesh : pyvista.PolyData
            The extracted and smoothed surface mesh ready for plotting.
        """
        img = nib.load(nii_fp)
        vol = img.get_fdata()

        if vol.ndim > 3:
            log.warning(f"[WARNING] detected {vol.ndim}d nifti volume. using the first volume (index 0).")
            vol = vol[..., 0]

        vol = np.nan_to_num(vol, nan=0.0)

        if blur_sigma and blur_sigma > 0:
            vol = gaussian_filter(vol, sigma=float(blur_sigma))

        mask = vol > float(threshold)

        if not np.any(mask):
            raise ValueError("Mask is empty after thresholding. Adjust threshold/blur_sigma.")

        verts_vox, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        verts_world = nib.affines.apply_affine(img.affine, verts_vox)

        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
        mesh = pv.PolyData(verts_world.astype(np.float32), faces_pv)

        if smooth_i and smooth_i > 0:
            mesh = mesh.smooth(n_iter=int(smooth_i), relaxation_factor=float(smooth_f))

        if mesh.n_points == 0:
            raise ValueError("Extracted mesh has no vertices. Check input mask and parameters.")

        try:
            mesh = mesh.fill_holes(1000)
        except Exception as exc:
            log.warning(f"Mesh hole filling failed: {exc}. Continuing with unfilled meshes.")

        return mesh

    @staticmethod
    def _crop_png_whitespace(png_path: str, white_threshold: int = 245, pad: int = 8) -> None:
        """Crop near-white borders from PNG previews to reduce empty space."""
        try:
            img = skio.imread(png_path)
            if img.size == 0:
                return

            if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(img)) <= 1.0:
                threshold = white_threshold / 255.0
            else:
                threshold = float(white_threshold)

            if img.ndim == 2:
                content_mask = img < threshold
            else:
                rgb = img[..., :3]
                content_mask = np.any(rgb < threshold, axis=-1)

            if not np.any(content_mask):
                return

            rows = np.where(np.any(content_mask, axis=1))[0]
            cols = np.where(np.any(content_mask, axis=0))[0]

            top = max(int(rows[0]) - pad, 0)
            bottom = min(int(rows[-1]) + pad + 1, img.shape[0])
            left = max(int(cols[0]) - pad, 0)
            right = min(int(cols[-1]) + pad + 1, img.shape[1])

            cropped = img[top:bottom, left:right]
            if cropped.shape != img.shape:
                skio.imsave(png_path, cropped, check_contrast=False)
        except Exception as exc:
            log.warning(f"Failed to crop white space for '{png_path}': {exc}")
