#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --context none --layer 2
python main.py --context low --layer 2
python main.py --context high --layer 2