import numpy
import random
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

from article_data import ArticleDataset

# Used for linear algebra calculations
import numpy as np

# Roberta base model is used
# for all classification models
from transformers import RobertaModel, LongformerModel

# Used to define the dataset and classifier
# objects
import torch
from torch import nn

# The Roberta classifiers use the Adam
# Optimizer
from torch.optim import Adam

# All models use a linear scheduler with warmup
from transformers import get_linear_schedule_with_warmup

# Used for output readability
from tqdm import tqdm

# Used to filter out zero division warnings
import warnings

warnings.filterwarnings('ignore')

# Used to filter out transformer warnings
from transformers import logging

logging.set_verbosity_error()

# Sklearn metrics are used to measure model performance
from sklearn.metrics import f1_score, classification_report
from copy import deepcopy
from utils import get_collate_fn
from utils import get_tokenizer

class RobertaClassifier(nn.Module):

  def __init__(self, config):
    super(RobertaClassifier, self).__init__()
    if config.longformer:
      self.roberta = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    else:
      self.roberta = RobertaModel.from_pretrained("roberta-base")

    if config.freeze_layers:
      for idx, (name,params) in enumerate(self.roberta.named_parameters()):
        if idx < config.freeze_layers:
          params.requires_grad = False
        # print(idx, name)
        # if name.startswith("encoder.layer") and int(name.split(".")[2]) > 7
    self.dropout = nn.Dropout(config.dropout)
    if config.classifier_second_layer:
      self.linear = nn.Linear(768, config.classifier_second_layer)
      torch.nn.init.xavier_uniform(self.linear.weight)  
      self.linear2 = nn.Linear(config.classifier_second_layer, config.num_labels)
      torch.nn.init.xavier_uniform(self.linear2.weight)
    else:
      self.linear = nn.Linear(768, config.num_labels)
      torch.nn.init.xavier_uniform(self.linear.weight)  
      self.linear2 = None

    self.relu = nn.ReLU()

  def forward(self, input_id, mask):
    output_1 = self.roberta(input_ids=input_id, attention_mask=mask)
    pooler = output_1.last_hidden_state[:, 0]
    dropout_output = self.dropout(pooler)
    linear_output = self.linear(dropout_output)
    if self.linear2:
      relu_layer = self.relu(linear_output)
      dropout_output2 = self.dropout(relu_layer)
      linear_output2 = self.linear2(dropout_output2)
      return linear_output2
    else:
      return linear_output


# Find the values used in weighted loss. The formula used is:
# ((1 - # of instances per class) / Total Instances)
def calculate_article_weights(data):
  counts = []
  for i in data["label"].value_counts()[::-1]:
    counts.append(i)

  num_instances = sum(counts)
  class_weights = 1 - (torch.tensor(counts, dtype=torch.float64) / num_instances)

  return class_weights

import sklearn
import os
# Training loop for the article modelling
def article_train(model, train_data, learning_rate,
                  epochs, batch_size, weights, column,
                  path,test_data):
  # Define the article dataset for training
  if os.path.exists(path):
    checkpoint = torch.load(path)
    best_model_weights = checkpoint["best_model_weights"]
    min_loss = checkpoint["min_loss"]
  else:
    best_model_weights = model.state_dict()
    min_loss = 10e10

  train_data, dev_data = sklearn.model_selection.train_test_split(train_data,test_size=0.1,random_state=2)
  print(calculate_article_weights(dev_data))
  train = ArticleDataset(train_data, column)

  # Define the train dataloader
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,collate_fn=get_collate_fn(), shuffle=True,worker_init_fn=seed_worker,generator=g)

  # Use GPU if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Define the loss function as weighted cross entropy
  criterion = nn.CrossEntropyLoss(weight=weights.float())
  # Define the optimizer as Adam
  optimizer = Adam(model.parameters(), lr=learning_rate)

  # Define the scheduler as linear with warmup
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=len(train_dataloader) * 100
  )

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(epochs):
    model.train()
    train_output = []
    train_targets = []
    train_loss = 0

    for train_input, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_input['attention_mask'].to(device)
      input_id = train_input['input_ids'].squeeze(1).to(device)
      model_batch_result = model(input_id, mask)

      batch_loss = criterion(model_batch_result, train_label.long())
      train_output.extend(np.argmax(model_batch_result.detach().cpu().numpy(), axis=1))
      train_targets.extend(train_label.detach().cpu().numpy())

      train_loss += batch_loss

      model.zero_grad()
      batch_loss.backward()
      optimizer.step()
      scheduler.step()
    print("epoch:{:2d} training: "
          "micro f1: {:.3f} "
          "macro f1: {:.3f} "
          "loss: {:.3f} ".format(epoch_num + 1,
                                 f1_score(y_true=train_targets,
                                          y_pred=train_output,
                                          average='micro',
                                          zero_division="warn"),
                                 f1_score(y_true=train_targets,
                                          y_pred=train_output,
                                          average='macro',
                                          zero_division="warn"),
                                 train_loss / len(train_data)))
    vaL_loss, val_micro_f1, val_macro_f1 = article_evaluate(model, dev_data, batch_size, column)
    print("TEST", end=" ")
    article_evaluate(model, test_data, batch_size, column)                    
    if vaL_loss < min_loss:
      min_loss = vaL_loss
      best_model_weights = deepcopy(model.state_dict())
      print("SAVED")
      torch.save({
        "min_loss": min_loss,
        "best_model_weights": deepcopy(model.state_dict())
        },
        path)



  return best_model_weights
def article_evaluate(model, test_data, batch_size, column,verbose=False,return_results=False):
  model.eval()
  test = ArticleDataset(test_data, column)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size,collate_fn=get_collate_fn(),worker_init_fn=seed_worker,generator=g)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  criterion = nn.CrossEntropyLoss()
  loss = 0
  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
  
  with torch.no_grad():
    test_output = []
    test_targets = []

    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)
      model_batch_result = model(input_id, mask)
      loss += criterion(model_batch_result, test_label.long())


      test_output.extend(np.argmax(model_batch_result.cpu().numpy(), axis=1))
      test_targets.extend(test_label.cpu().numpy())

  labels = ["True", "False"]
  macro_f1 = f1_score(y_true=test_targets,
                                      y_pred=test_output,
                                      average='macro',
                                      zero_division="warn")
  micro_f1 = f1_score(y_true=test_targets,
                                      y_pred=test_output,
                                      average='micro',
                                      zero_division="warn")

  if verbose:
    print(classification_report(test_targets, test_output, target_names=labels, digits=3))
  print("Validate micro f1: {:.3f} "
      "macro f1: {:.3f} "
      "loss: {:.3f}"
      .format(micro_f1,macro_f1,loss))
  if return_results:
    return test_output
  return loss, micro_f1, macro_f1

