from anno_file_prep import prepare_files
from segment_data import construct_segment_dict
from segment_data import convert_to_dataframe
from strategy_model import calculate_weights
from strategy_model import strategy_evaluate
from detection_model import RobertaClassifier
from configs import get_config
import numpy as np
import torch
import os
import torch
import argparse
from utils import set_seed
SPLIT = 0.8

parser = argparse.ArgumentParser()
parser.add_argument('--models_address', type=str,default="./models")
args = parser.parse_args()
layers = ["1","2","3","4"]
contexts = ["none", "low", "high"]


for layer in layers:
    for context in contexts:
        # Create the train annotation dictionary
        train_dict = construct_segment_dict("data/anno_train", context)

        # Create the dataframes for all 4 layers of classification,
        # along with their weights
        train = convert_to_dataframe(train_dict, layer=layer)
        weights = calculate_weights(train)
        # Create the test annotation dictionary
        test_dict = construct_segment_dict("data/anno_test", context)

        # Create the test dataframes
        test = convert_to_dataframe(test_dict, layer=layer)
        config = get_config(layer,context)
        model_name_wild_card =  f"{layer}_{context}"
        macro_f1s = []
        micro_f1s = []
        model = []
        max_f1 = 0
        import re
        for model_address in filter(lambda x:re.match(f"{layer}_{context}_[0-9]+", x),os.listdir(path=args.models_address)):
            path = os.path.join(args.models_address, model_address)
            print(path)
            model = RobertaClassifier(config)
            loaded = torch.load(path)
            weights = loaded["weights"]
            max_val_acc = loaded["max_val_acc"]
            model.load_state_dict(weights)
            # Create the Roberta strategy models, the output length being the number
            # of labels in that layer
            macro_f1, micro_f1 = strategy_evaluate(model=model, test_data=test, context=context, layer=layer,config=config,verbose=False)
            if macro_f1 > max_f1:
                max_f1 =  macro_f1
                torch.save(loaded, os.path.join(args.models_address, f"{layer}_{context}_best.pt"))
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        micro_f1s = np.array(micro_f1s)
        macro_f1s = np.array(macro_f1s)
        print("####","Average RESULT",context, layer, "micro", f"{micro_f1s.mean():.3f}", "macro",f"{ macro_f1s.mean():.3f}")

        loaded = torch.load(os.path.join(args.models_address, f"{layer}_{context}_best.pt"))
        weights = loaded["weights"]
        max_val_acc = loaded["max_val_acc"]
        macro_f1, micro_f1 = strategy_evaluate(model=model, test_data=test, context=context, layer=layer,config=config,verbose=True)
