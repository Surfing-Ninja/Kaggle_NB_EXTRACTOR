#!/bin/bash
# Convenience script to run Python commands with the virtual environment

cd "$(dirname "$0")"
./venv/bin/python "$@"
