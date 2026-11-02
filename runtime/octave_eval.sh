#!/bin/sh
exec /opt/anaconda3/bin/conda run -n visdrone-eval --no-capture-output octave "$@"
