"""Convert SVG figures for builders that require PNG."""

import shutil
import subprocess

from sphinx.errors import ExtensionError
from sphinx.transforms.post_transforms.images import ImageConverter


class SvgToPng(ImageConverter):
    conversion_rules = [("image/svg+xml", "image/png")]

    def is_available(self):
        if shutil.which("rsvg-convert") is None:
            raise ExtensionError("Install rsvg-convert to build the guide PDF.")
        return True

    def convert(self, source, destination):
        try:
            subprocess.run(
                ["rsvg-convert", "--width=1600", "--background-color=white",
                 "--output", destination, source],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ExtensionError(f"Could not convert {source}: {exc}") from exc
        return True


def setup(app):
    app.add_post_transform(SvgToPng)
    return {"parallel_read_safe": True}
