#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python main.py --context none --layer 4
# python main.py --context none --layer 4
# python main.py --context none --layer 4
# python main.py --context low --layer 4
# python main.py --context low --layer 4
# python main.py --context low --layer 4
python main.py --context high --layer 4
# python main.py --context high --layer 4
python main.py --context high --layer 4