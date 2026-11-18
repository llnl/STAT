# Documentation

The STAT User Guide uses the Markdown sources in `doc/src/`.

## Build with Docker

Run the same build as GitHub Actions from the repository root.

```sh
docker build -t stat-user-guide doc
mkdir -p doc/_build/site
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/doc/_build/site:/output" stat-user-guide
```

Open `doc/_build/site/index.html` for the HTML guide or
`doc/_build/site/stat_userguide.pdf` for the PDF.

## Setup

Building the guide requires Python 3.11 or later. Run the commands from the
repository root.

### macOS

```sh
brew install python@3.13
python3.13 -m venv doc/venv

# For PDF output, install Tectonic.
brew install tectonic
```

### Ubuntu 24.04

Use Ubuntu's default Python 3.12:

```sh
sudo apt update
sudo apt install python3-venv
python3 -m venv doc/venv

# For PDF output, install XeLaTeX and its build tools.
sudo apt install --no-install-recommends make latexmk texlive-xetex texlive-fonts-recommended fonts-freefont-otf
```

### Activate the environment on either platform

```sh
source doc/venv/bin/activate
python -m pip install -r doc/requirements.txt
```

## Build the user guide

```sh
sphinx-build -M html doc/src doc/_build/userguide -c doc/userguide -W --keep-going -n
```

For PDF output on Ubuntu:

```sh
sphinx-build -M latexpdf doc/src doc/_build/userguide -c doc/userguide -W --keep-going -n
```

For PDF output on macOS:

```sh
sphinx-build -M latex doc/src doc/_build/userguide -c doc/userguide -W --keep-going -n
(cd doc/_build/userguide/latex && tectonic stat_userguide.tex)
```

Open the generated files on macOS:

```sh
open doc/_build/userguide/html/index.html
open doc/_build/userguide/latex/stat_userguide.pdf
```

## Local web preview

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory doc/_build/userguide/html
```

Visit <http://127.0.0.1:8000/>. Press Ctrl-C to stop.
