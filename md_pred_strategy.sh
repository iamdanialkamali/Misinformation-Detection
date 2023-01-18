#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python misinformation_detection.py --source pred_strategy --context none
python misinformation_detection.py --source pred_strategy --context none
python misinformation_detection.py --source pred_strategy --context none

python misinformation_detection.py --source pred_strategy --context low
python misinformation_detection.py --source pred_strategy --context low
python misinformation_detection.py --source pred_strategy --context low

python misinformation_detection.py --source pred_strategy --context high
python misinformation_detection.py --source pred_strategy --context high
python misinformation_detection.py --source pred_strategy --context high