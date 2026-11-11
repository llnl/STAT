#!/usr/bin/env bash

# this script is sourced by tests after installation to use STAT.

export STAT_CI_ROOT="${STAT_CI_ROOT:-/opt}"
export STAT_PIXI="$STAT_CI_ROOT/stat-pixi"
export STAT_DEPS="$STAT_CI_ROOT/stat-deps"
export PIXI_PREFIX="$STAT_PIXI/.pixi/envs/default"

export PATH="$STAT_DEPS/bin:$PIXI_PREFIX/bin:$STAT_CI_ROOT/pixi/bin:$PATH"
export LD_LIBRARY_PATH="$STAT_DEPS/lib:$PIXI_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$STAT_DEPS/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
