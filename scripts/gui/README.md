# Standalone STAT GUI

The standalone Python GUI can be installed without building STAT, Dyninst, LaunchMON, MRNet or GraphLib. It provides the
basis for remote GUI functionality planned for future changes.

## Installing the GUI and its dependencies

Python 3.10 or newer, GTK 3, PyGObject, PyCairo and the Graphviz `dot` executable are required as core dependencies. They can
come from system packages or environments such as Homebrew and Pixi. Pip installs xdot and Pygments and their Python
requirements.

The standalone installer creates a Pixi environment and installs the STAT GUI and all its dependencies. The following example
uses a local STAT checkout on Linux or macOS, after cloning or updating the desired branch:

```sh
# Install from the local checkout
bash scripts/gui/install.sh --source "$PWD" --location "$HOME/stat-gui"

# Start stat-gui
pixi run --manifest-path "$HOME/stat-gui/pixi/pixi.toml" stat-gui
```

The installer copies `scripts/gui/pixi.toml` into the selected installation directory and creates an isolated Pixi
environment with all dependencies. It checks GTK imports and command-line help without opening a window. It prints launch
commands that can be run on a graphical desktop.

The default installation location is `$STAT_HOME` when set, otherwise `$HOME/stat-gui`. If Pixi is unavailable,
the script installs it under the selected location.

An existing installation can use a new version or updated local code while reusing the installed dependencies:

```sh
pixi run --manifest-path "$HOME/stat-gui/pixi/pixi.toml" \
  python -m pip install --no-deps --force-reinstall /path/to/STAT/scripts/gui
```

## Ubuntu: Installing with pip using system packages

On Debian/Ubuntu, the graphics prerequisites are available as system packages. These only need to be installed once:

```sh
sudo apt install python3-venv python3-gi python3-gi-cairo gir1.2-gtk-3.0 graphviz librsvg2-common
```

A virtual environment with access to the system GTK bindings can then host the GUI. From a STAT checkout, the installation
looks like this:

```sh
python3 -m venv --system-site-packages .venv-stat-gui
. .venv-stat-gui/bin/activate
python -m pip install ./scripts/gui
stat-view
```

For an environment that already provides GTK 3, PyGObject, PyCairo and Graphviz, only `python -m pip install ./scripts/gui`
is needed.

## macOS: Installing with Pixi and Homebrew

The same Pixi installation instructions used on Linux apply on macOS; follow
[Installing the GUI and its dependencies](#installing-the-gui-and-its-dependencies) above.

To use [Homebrew](https://brew.sh/), install Python and the graphics dependencies:

```sh
brew install python pygobject3 gtk+3 graphviz
```

From the root of a STAT checkout, create a virtual environment using Homebrew's Python. The
`--system-site-packages` option makes Homebrew's PyGObject and PyCairo bindings available inside it:

```sh
"$(brew --prefix)/bin/python3" -m venv --system-site-packages .venv-stat-gui
. .venv-stat-gui/bin/activate
python -m pip install ./scripts/gui

# Check the graphics bindings and command-line entry points
python -m pip check
python -c 'import gi, cairo; gi.require_version("Gtk", "3.0"); from gi.repository import Gtk; print("GTK", Gtk.get_major_version(), Gtk.get_minor_version())'
stat-gui --version

# Open the viewer, optionally with a saved STAT call graph
stat-gui
stat-view /path/to/saved.2D.dot
```

Homebrew must be on `PATH` so the viewer can find Graphviz's `dot`. Use the Homebrew Python matching the installed
`pygobject3` bindings; a different Python installation or minor version may fail to import `gi` or `cairo`.


## Installing from a Git repository

Pip can clone a repository and install only this package subdirectory:

```sh
python -m pip install "git+https://github.com/LLNL/STAT.git@develop#subdirectory=scripts/gui"
```

The example uses `LLNL/STAT` and the `develop` branch. These can be replaced with another repository and branch containing
this package. Installation from a Git URL requires Git and the same graphics prerequisites as a local pip installation.


## Opening saved call graphs

```sh
stat-view /path/to/saved.2D.dot
stat-gui
stat-gui --version
```

SSH GUI/server connections and remote functionality are not included in this version.
