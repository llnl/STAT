"""Launch the STAT GUI with native or remote collection.

Prepare GTK resources before importing viewer modules. Offline help and version
reporting need neither GTK nor a display. Console commands are registered in
pyproject.toml.
"""

import argparse
import atexit
import glob
from importlib.metadata import version
import os
import runpy
import shutil
import subprocess
import sys
import tempfile


def _has_file_chooser_schema(schema_dir):
    """Check the candidate's compiled schemas without inheriting global ones."""
    try:
        from gi.repository import Gio, GLib
    except ImportError:
        return False

    try:
        source = Gio.SettingsSchemaSource.new_from_directory(schema_dir, None, False)
        return source.lookup('org.gtk.Settings.FileChooser', False) is not None
    except GLib.Error:
        return False


def _configure_gsettings_schema_dir():
    """Find valid GTK schemas in environment prefixes before GLib's defaults.

    Relocated Pixi/Conda environments can otherwise abort on opening a chooser.
    """
    # Prefer explicit settings and active environments over platform defaults.
    candidates = [os.environ.get('GSETTINGS_SCHEMA_DIR')]
    for prefix in (os.environ.get('CONDA_PREFIX'), sys.prefix,
                   os.environ.get('HOMEBREW_PREFIX')):
        if prefix:
            candidates.append(os.path.join(prefix, 'share', 'glib-2.0', 'schemas'))
    if sys.platform == 'darwin':
        candidates.extend([
            '/opt/homebrew/share/glib-2.0/schemas',
            '/usr/local/share/glib-2.0/schemas',
        ])

    for schema_dir in candidates:
        if not schema_dir:
            continue
        # Repair a missing schema cache when package setup hooks were skipped.
        compiled = os.path.join(schema_dir, 'gschemas.compiled')
        if not os.path.isfile(compiled):
            schema = os.path.join(schema_dir, 'org.gtk.Settings.FileChooser.gschema.xml')
            compiler = shutil.which('glib-compile-schemas')
            if not os.path.isfile(schema) or compiler is None:
                continue
            try:
                subprocess.run([compiler, schema_dir], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                continue
        # Accept only schemas provided by this directory, without global fallback.
        if _has_file_chooser_schema(schema_dir):
            os.environ['GSETTINGS_SCHEMA_DIR'] = schema_dir
            return


def _configure_gdk_pixbuf_loaders():
    """Register bundled image loaders when Pixi's post-link hook was skipped."""
    if os.environ.get('GDK_PIXBUF_MODULE_FILE'):
        return

    # Locate the loader modules and query tool from the same runtime prefix.
    prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
    loader_dir = os.path.join(prefix, 'lib', 'gdk-pixbuf-2.0', '2.10.0', 'loaders')
    loaders = [path for path in glob.glob(os.path.join(loader_dir, '*'))
               if os.path.isfile(path)]
    query_loaders = os.path.join(prefix, 'bin', 'gdk-pixbuf-query-loaders')
    if not loaders or not os.path.isfile(query_loaders):
        return

    # Use a private per-process cache, avoiding stale paths after package updates.
    try:
        cache = subprocess.check_output([query_loaders] + loaders, text=True, timeout=10)
        cache_dir = tempfile.TemporaryDirectory(prefix='stat-gui-pixbuf-')
        atexit.register(cache_dir.cleanup)
        cache_path = os.path.join(cache_dir.name, 'loaders.cache')
        with open(cache_path, 'w') as cache_file:
            # Pixi can emit prefix-relative paths; the temporary cache needs absolute ones.
            cache_file.write(cache.replace('"lib/', '"%s/lib/' % prefix))
        os.environ['GDK_PIXBUF_MODULE_FILE'] = cache_path
    except (OSError, subprocess.SubprocessError):
        # GTK can still start without optional SVG theme assets.
        pass


def _set_application_identity():
    """Set the desktop-visible name before GTK creates the application."""
    try:
        from gi.repository import GLib
        GLib.set_prgname('STAT-GUI')
        GLib.set_application_name('STAT-GUI')
    except (ImportError, AttributeError):
        pass


def _main(command):
    """Prepare graphics resources and dispatch to the GUI or viewer."""

    # Version reporting needs neither GTK initialization nor a display.
    if sys.argv[1:] == ['--version']:
        print('STAT-GUI %s' % version('stat-gui'))
        return

    if command == 'view':
        parser = argparse.ArgumentParser(
            description='View STAT call graphs without a native STAT installation.')
        parser.add_argument('files', nargs='*', help='optional list of STAT .dot files')
        args = parser.parse_args()

    elif '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        from remote.stat_client_gui import gui_help
        gui_help()
        return

    # Finish environment preparation before importing GTK-dependent modules.
    _configure_gsettings_schema_dir()
    _configure_gdk_pixbuf_loaders()
    _set_application_identity()

    # Dispatch to the existing viewer or native GUI implementation.
    if command == 'view':
        from STATview import STATview_main
        STATview_main(args)
    else:
        sys.argv.insert(1, command)
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stat_main = os.path.join(module_dir, 'STATmain.py')
        runpy.run_path(stat_main, run_name='__main__')


def main():
    """Open the GUI with local bindings or remote SSH collection."""
    _main('gui')


def view_main():
    """Open saved call graphs without importing native collection modules."""
    _main('view')
