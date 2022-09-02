from article_data import ArticleDataset

# Used for linear algebra calculations
import numpy as np

# Roberta base model is used
# for all classification models
from transformers import RobertaModel

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


class RobertaClassifier(nn.Module):

  def __init__(self, output_length, dropout=0.5):
    super(RobertaClassifier, self).__init__()

    self.roberta = RobertaModel.from_pretrained("roberta-base")
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(768, 1024)
    self.linear2 = nn.Linear(1024, output_length)
    self.relu = nn.ReLU()

  def forward(self, input_id, mask):
    output_1 = self.roberta(input_ids=input_id, attention_mask=mask)
    hidden_state = output_1[0]
    pooler = hidden_state[:, 0]
    dropout_output = self.dropout(pooler)
    linear_output = self.linear(dropout_output)
    relu_layer = self.relu(linear_output)
    dropout_output2 = self.dropout(relu_layer)
    linear_output2 = self.linear2(dropout_output2)
    return linear_output2


# Find the values used in weighted loss. The formula used is:
# ((1 - # of instances per class) / Total Instances)
def calculate_article_weights(data):
  counts = []
  for i in data["label"].value_counts()[::-1]:
    counts.append(i)

  num_instances = sum(counts)
  class_weights = 1 - (torch.tensor(counts, dtype=torch.float64) / num_instances)

  return class_weights


# Training loop for the article modelling
def article_train(model, train_data, learning_rate,
                  epochs, batch_size, weights, column):

  # Define the article dataset for training
  train = ArticleDataset(train_data, column)

  # Define the train dataloader
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

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
    num_training_steps=len(train_dataloader) * epochs
  )

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(epochs):

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


def article_evaluate(model, test_data, batch_size, column):

  test = ArticleDataset(test_data, column)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  if use_cuda:
    model = model.cuda()

  with torch.no_grad():
    test_output = []
    test_targets = []

    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      model_batch_result = model(input_id, mask)
      test_output.extend(np.argmax(model_batch_result.cpu().numpy(), axis=1))
      test_targets.extend(test_label.cpu().numpy())

  labels = ["True", "False"]
  print(classification_report(test_targets, test_output, target_names=labels, digits=3))
