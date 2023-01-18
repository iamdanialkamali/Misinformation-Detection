import os
import pandas as pd
import torch
import argparse
import random
from segment_data import construct_segment_dict
from segment_data import convert_to_dataframe
from strategy_model import strategy_evaluate
from detection_model import RobertaClassifier
from detection_model import article_train
from detection_model import article_evaluate
from detection_model import calculate_article_weights
from article_data import build_complete_dataset
from configs import get_config
from utils import set_seed
import numpy as np
class Config:
    dropout = 0.5
    max_length = 512
    num_labels = 2
    freeze_layers = 181
    longformer = False
    classifier_second_layer = None
    batch_size = 10
    epochs = 50
    learning_rate = 5e-5
    seed = random.randint(0,99999999)
    # seed = 9877

# Declare the learning rate and batch size for detection training
article_config = Config()
set_seed(article_config.seed)


test_data = pd.read_csv(f"data/test_article_data_none.csv")
corr_labels = {True: "true", False: "false"}
test_data["label"] = test_data["label"].apply(lambda x: corr_labels[x])
test_data = test_data[test_data["label"] != "none"]

# Define the layer to train and test on.
# "article" for base article text
# "target_combined" for article and ground truth labels
# "pred_combined" for article and predicted labels

models_address= "./models/"
import re
for column in ["claim","article","gt_strategy", "claim_article","pred_strategy","claim_gt","claim_article_gt" ,"claim_pred","claim_article_pred"]:
    for model_address in filter(lambda x:re.match(f"{column}\_none_best", x),os.listdir(path=models_address)):
        path = os.path.join(models_address, model_address)
        print(path)
        strategy_model = RobertaClassifier(article_config)
        loaded = torch.load(path)
        best_model_weights = loaded["best_model_weights"]
        strategy_model.load_state_dict(best_model_weights)
        res = article_evaluate(model=strategy_model, test_data=test_data, batch_size=article_config.batch_size, column=column, verbose=False,return_results=True)
        test_data[model_address.split(".")[0]] = [str(not bool(x)).lower() for x in res]
test_data.to_csv("data/test_article_analysis.csv")