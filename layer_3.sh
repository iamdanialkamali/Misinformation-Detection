#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --context none --layer 3
python main.py --context none --layer 3
python main.py --context none --layer 3
python main.py --context low --layer 3
python main.py --context low --layer 3
python main.py --context low --layer 3
python main.py --context high --layer 3
python main.py --context high --layer 3
python main.py --context high --layer 3