#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python misinformation_detection_test.py --source article
python misinformation_detection_test.py --source claim
python misinformation_detection_test.py --source gt_strategy
python misinformation_detection_test.py --source pred_strategy --context none
python misinformation_detection_test.py --source pred_strategy --context low
python misinformation_detection_test.py --source pred_strategy --context high
python misinformation_detection_test.py --source claim_article
python misinformation_detection_test.py --source claim_article_gt
python misinformation_detection_test.py --source claim_article_pred --context none
python misinformation_detection_test.py --source claim_article_pred --context low
python misinformation_detection_test.py --source claim_article_pred --context high
python misinformation_detection_test.py --source claim_gt
python misinformation_detection_test.py --source claim_pred --context none
python misinformation_detection_test.py --source claim_pred --context low
python misinformation_detection_test.py --source claim_pred --context high
