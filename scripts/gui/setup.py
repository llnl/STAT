"""Build the standalone GUI from this directory or a self-contained archive.

The native Autotools build owns the shared GUI modules one directory above.
Copy only those selected modules into Python builds instead of duplicating the
sources in this package. Source archives also receive these modules, the logo
and license, so wheel rebuilds need no enclosing STAT checkout.
"""

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


SOURCE_DIR = Path(__file__).resolve().parent
SHARED_MODULES = ('STATmain.py', 'STATGUI.py', 'STATview.py', 'STAThelper.py')
CLIENT_MODULES = (
    '__init__.py',
    'stat_client_cli.py',
    'stat_client_connection.py',
    'stat_ssh.py',
    'stat_client_gui.py',
    'stat_client_slurm.py',
    'stat_server_slurm.py',
)
RESOURCES = ('STATlogo.gif', 'LICENSE')


def shared_path(name):
    """Locate a shared Python module in an archive or the native source tree."""
    local = SOURCE_DIR / name
    return local if local.is_file() else SOURCE_DIR.parent / name


def resource_path(name):
    """Locate a desktop resource in an archive or the repository root."""
    local = SOURCE_DIR / name
    return local if local.is_file() else SOURCE_DIR.parent.parent / name


def client_path(name):
    local = SOURCE_DIR / 'remote' / name
    return local if local.is_file() else SOURCE_DIR.parent / 'remote' / name


class BuildPy(build_py):
    """Bundle shared modules without changing their native source locations."""

    def run(self):
        super().run()
        remote = Path(self.build_lib) / 'remote'
        remote.mkdir(exist_ok=True)
        for name in CLIENT_MODULES:
            self.copy_file(str(client_path(name)), str(remote / name))
        for name in SHARED_MODULES:
            self.copy_file(str(shared_path(name)), str(Path(self.build_lib) / name))


class SDist(sdist):
    """Include shared sources and resources in a self-contained archive."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        for name in SHARED_MODULES:
            shutil.copyfile(shared_path(name), Path(base_dir) / name)
        remote = Path(base_dir) / 'remote'
        remote.mkdir(exist_ok=True)
        for name in CLIENT_MODULES:
            shutil.copyfile(client_path(name), remote / name)
        for name in RESOURCES:
            shutil.copyfile(resource_path(name), Path(base_dir) / name)


setup(
    cmdclass={'build_py': BuildPy, 'sdist': SDist},
    data_files=[('share/stat', [str(resource_path(name)) for name in RESOURCES])],
)
