#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python misinformation_detection.py --source claim_article_pred --context none
python misinformation_detection.py --source claim_article_pred --context none
python misinformation_detection.py --source claim_article_pred --context none

python misinformation_detection.py --source claim_article_pred --context low
python misinformation_detection.py --source claim_article_pred --context low
python misinformation_detection.py --source claim_article_pred --context low

python misinformation_detection.py --source claim_article_pred --context high
python misinformation_detection.py --source claim_article_pred --context high
python misinformation_detection.py --source claim_article_pred --context high
