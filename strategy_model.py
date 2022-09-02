# Roberta tokenizer is used for classification models
from transformers import RobertaTokenizer

# Used to define the dataset and classifier objects
import torch
from torch import nn

# Used for linear algebra calculations
import numpy as np

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

from segment_data import SegmentDataset


# Find the values used in weighted loss. The formula used is:
# ((1 - # of instances per class) / Total Instances)
def calculate_weights(data):
  counts = []
  if len(data.columns[1:]) == 1:
    for i in data["hasAnno"].value_counts()[::-1]:
      counts.append(i)
  else:
    for i in data.columns[1:]:
      counts.append(sum(data[i].values))

  num_instances = sum(counts)
  class_weights = 1 - (torch.tensor(counts, dtype=torch.float64) / num_instances)

  return class_weights


# Edit the predicted output into a workable format
def calculate_metrics(pred, layer, threshold=0.5):

  # If layer 1, change the prediction output to be the argmax
  if layer == "1":
    new_pred = []
    for prediction in pred:
      new_pred.append(np.argmax(prediction))
    pred = new_pred

  # If any other layer, binarize the output using the threshold.
  elif layer in ["2", "3", "4"]:
    pred = np.array(pred > threshold, dtype=float)

  return pred


def strategy_train(model, train_data, learning_rate, epochs,
                  batch_size, context, layer, class_weights):

  # Define the segment dataset for training
  train = SegmentDataset(train_data, context)

  # Define the train dataloader
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

  # Use GPU if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Define the loss function as weighted cross entropy, or binary cross entropy
  if layer == "1":
    criterion = nn.CrossEntropyLoss(weight=class_weights.float())
  elif layer in ["2", "3", "4"]:
    criterion = nn.BCEWithLogitsLoss(weight=class_weights.float())
  else:
    raise Exception("Invalid Layer given (must be '1', '2', '3' or '4')")

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

      if layer == "1":
        batch_loss = criterion(model_batch_result.float(), train_label.long().squeeze(1))
      else:
        batch_loss = criterion(model_batch_result.float(), train_label)

      train_output.extend(model_batch_result.detach().cpu().numpy())
      train_targets.extend(train_label.detach().cpu().numpy())

      train_loss += batch_loss

      model.zero_grad()
      batch_loss.backward()
      optimizer.step()
      scheduler.step()

    train_pred = calculate_metrics(np.array(train_output), layer)
    print("epoch:{:2d} training: "
          "micro f1: {:.3f} "
          "macro f1: {:.3f} "
          "loss: {:.3f} ".format(epoch_num + 1,
                                 f1_score(y_true=train_targets,
                                          y_pred=train_pred,
                                          average='micro',
                                          zero_division="warn"),
                                 f1_score(y_true=train_targets,
                                          y_pred=train_pred,
                                          average='macro',
                                          zero_division="warn"),
                                 train_loss / len(train_data)))


def strategy_evaluate(model, test_data, context, layer):
  test = SegmentDataset(test_data, context)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)

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
      test_output.extend(model_batch_result.cpu().numpy())
      test_targets.extend(test_label.long().squeeze(1).cpu().numpy())

  pred = calculate_metrics(np.array(test_output), layer)

  if layer == "1":
    key_to_label = {
      0: 'No Annotation',
      1: 'Has Annotation'
    }
  elif layer == "2":
    key_to_label = {
      0: 'Narrative w/ Details',
      1: 'Personal Exp. as Evidence',
      2: 'Distrusting Govt.',
      3: 'Politicizing health',
      4: 'Highlighting uncertainty/risk',
      5: 'Exploiting Sciences\' limits',
      6: 'Inapt. Use of Scientific Evidence',
      7: 'Rhetorical Tricks',
      8: 'Biased Reasoning to make conclusion',
      9: 'Emotional Appeals',
      10: 'Dist. Language Features',
      11: 'Establishing Legitimacy'
    }
  elif layer == "3":
    key_to_label = {
      0: 'Verified to be false',
      1: 'Verified to be true',
      2: 'Not Verified',
      3: 'Financial Motive',
      4: 'Freedom of choice and agency',
      5: 'Ingroup vs. Outgroup',
      6: 'Political figures/argument',
      7: 'Religion/Ideology',
      8: 'Out of context-Verified',
      9: 'Less Robust/Outdated Evidence-Verify',
      10: 'Lack of Citation for Evidence',
      11: 'Exaggeration/Absolute Language',
      12: 'Inapproprate Analogy of False Connection',
      13: 'Wrong Cause/Effect',
      14: 'Lack of Evidence or Incomplete Evidence',
      15: 'Evidence doesn\'t support conclusion',
      16: 'Shifting Hypothesis',
      17: 'Fear',
      18: 'Anger',
      19: 'Hope',
      20: 'Anxiety',
      21: 'Uppercase Words',
      22: 'Linguistic Intensifier',
      23: 'Clickbait Title',
      24: 'Bolded, underlined or Italicized',
      25: 'Exaggerated Punctuation',
      26: 'Citing Source to establish Legitimacy',
      27: 'Legitimate Persuasive Techniques',
      28: 'Surface Credibility Markers',
      29: 'Call to Action'
    }
  elif layer == "4":
    key_to_label = {
      0: 'source verified to be credible',
      1: 'source verified to not be credible',
      2: 'source not verified',
      3: 'source verified to be made up',
      4: 'rhetorical question',
      5: 'humor',
      6: 'medical or scientific jargon',
      7: 'words associated with nature or healthiness',
      8: 'simply claiming authority or credibility'
    }

  labels = []
  label_set = set()
  if layer == "1":
    labels = ["NoAnno", "HasAnno"]
  else:
    for i in test_targets:
      winner = np.argwhere(i == np.amax(i))
      for j in winner.flatten().tolist():
        label_set.add(j)
    for i in pred:
      winner = np.argwhere(i == np.amax(i))
      for j in winner.flatten().tolist():
        label_set.add(j)
    for i in sorted(label_set):
      labels.append(key_to_label[i])

  print(classification_report(test_targets, pred, target_names=labels, digits=3))


def strat_pred(model, sequence, context):
  device = torch.device("cuda")
  model = model.to(device)
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  context_to_len = {"none": 128, "low": 192, "high": 256}
  try:
    seq_length = context_to_len[context]
  except:
    raise "Invalid Context given (must be 'none', 'low', or 'high')"

  tokens = tokenizer(sequence, padding='max_length',
                     max_length=seq_length,
                     truncation=True, return_tensors="pt")
  input = tokens["input_ids"].squeeze(1).to(device)
  mask = tokens["attention_mask"].to(device)
  result = model(input, mask)

  return result.detach().cpu().numpy()[0]
