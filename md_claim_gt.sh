#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python misinformation_detection.py --source claim_gt
python misinformation_detection.py --source claim_gt
python misinformation_detection.py --source claim_gt