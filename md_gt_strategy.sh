#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python misinformation_detection.py --source gt_strategy
python misinformation_detection.py --source gt_strategy
python misinformation_detection.py --source gt_strategy