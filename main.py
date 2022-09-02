from anno_file_prep import prepare_files
from segment_data import construct_segment_dict
from segment_data import convert_to_dataframe
from strategy_model import calculate_weights
from strategy_model import strategy_train
from strategy_model import strategy_evaluate
from detection_model import RobertaClassifier
from detection_model import article_train
from detection_model import article_evaluate
from detection_model import calculate_article_weights
from article_data import build_complete_dataset

import pandas as pd

# # Define the annotation file location
# filename = 'data/annotation'
#
# # Define the train/test split and prepare the annotation files
# # This will create data/anno_train and data/anno_test
# split = 0.8
# prepare_files(filename, split)

# Level of context is used in constructing annotation dictionaries,
# declaring strategy models, and creating article datasets
context = "high"

# Create the train annotation dictionary
train_dict = construct_segment_dict("data/anno_train", context)

# Create the dataframes for all 4 layers of classification,
# along with their weights
layer1_train = convert_to_dataframe(train_dict, layer="1")
layer1_weights = calculate_weights(layer1_train)
layer2_train = convert_to_dataframe(train_dict, layer="2")
layer2_weights = calculate_weights(layer2_train)
layer3_train = convert_to_dataframe(train_dict, layer="3")
layer3_weights = calculate_weights(layer3_train)
layer4_train = convert_to_dataframe(train_dict, layer="4")
layer4_weights = calculate_weights(layer4_train)

# Create the test annotation dictionary
test_dict = construct_segment_dict("data/anno_test", context)

# Create the test dataframes
layer1_test = convert_to_dataframe(test_dict, layer="1")
layer2_test = convert_to_dataframe(test_dict, layer="2")
layer3_test = convert_to_dataframe(test_dict, layer="3")
layer4_test = convert_to_dataframe(test_dict, layer="4")

# Create the Roberta strategy models, the output length being the number
# of labels in that layer
layer1_model = RobertaClassifier(output_length=2)
layer2_model = RobertaClassifier(output_length=12)
layer3_model = RobertaClassifier(output_length=30)
layer4_model = RobertaClassifier(output_length=9)

# Declare the learning rate and batch size for strategy training
learning_rate = 5e-5
batch_size = 16

# Train and evaluate the 4 layer strategy models
strategy_train(model=layer1_model, train_data=layer1_train, learning_rate=learning_rate, epochs=4,
               batch_size=batch_size, context=context, layer="1", class_weights=layer1_weights)
strategy_evaluate(model=layer1_model, test_data=layer1_test, context=context, layer="1")
strategy_train(model=layer2_model, train_data=layer2_train, learning_rate=learning_rate, epochs=10,
               batch_size=batch_size, context=context, layer="2", class_weights=layer2_weights)
strategy_evaluate(model=layer2_model, test_data=layer2_test, context=context, layer="2")
strategy_train(model=layer3_model, train_data=layer3_train, learning_rate=learning_rate, epochs=20,
               batch_size=batch_size, context=context, layer="3", class_weights=layer3_weights)
strategy_evaluate(model=layer3_model, test_data=layer3_test, context=context, layer="3")
strategy_train(model=layer4_model, train_data=layer4_train, learning_rate=learning_rate, epochs=10,
               batch_size=batch_size, context=context, layer="4", class_weights=layer4_weights)
strategy_evaluate(model=layer4_model, test_data=layer4_test, context=context, layer="4")

# Create the train article dataset (This takes a very long time)
train_article_data = build_complete_dataset("data/anno_train",
                                            [layer1_model, layer2_model, layer3_model, layer4_model],
                                            context)
train_article_data.to_csv("data/train_article_data.csv")

# Create the test article dataset (This takes a very long time)
test_article_data = build_complete_dataset("data/anno_test",
                                           [layer1_model, layer2_model, layer3_model, layer4_model],
                                           context)
test_article_data.to_csv("data/test_article_data.csv")

# # Read the train and test article datasets into dataframes
# train_data = pd.read_csv("data/train_article_data.csv")
# train_data = train_data[train_data["label"] != "none"]
#
# test_data = pd.read_csv("data/test_article_data.csv")
# corr_labels = {True: "true", False: "false"}
# test_data["label"] = test_data["label"].apply(lambda x: corr_labels[x])
# test_data = test_data[test_data["label"] != "none"]
#
# # Calculate the weights based on the training set
# weights = calculate_article_weights(train_data)
#
# # Define the fake news detection model
# strategy_model = RobertaClassifier(output_length=2)
#
# # Declare the learning rate and batch size for detection training
# learning_rate = 2e-5
# batch_size = 4
#
# # Define the layer to train and test on.
# # "article" for base article text
# # "target_combined" for article and ground truth labels
# # "pred_combined" for article and predicted labels
# column = "HC_pred_combined"
#
# # Train and test the detection model
# article_train(model=strategy_model, train_data=train_data, learning_rate=learning_rate,
#               epochs=6, batch_size=batch_size, weights=weights, column=column)
# article_evaluate(model=strategy_model, test_data=test_data, batch_size=batch_size, column=column)