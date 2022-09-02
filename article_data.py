import os

# Used to read json files
import json

# Used to work with textual data
import re

# Used for linear algebra calculations
import numpy as np

# Used for dataframes
import pandas as pd

# Roberta tokenizer and base model are used
# for all classification models
from transformers import RobertaTokenizer

# Importing package and summarizer
import gensim
from gensim.summarization import summarize

# Used for suppressing warnings
from IPython.display import clear_output

# Used to tokenize articles into sentences
import nltk
from nltk import tokenize
nltk.download('punkt')

# Used to define the dataset and classifier
# objects
import torch

from strategy_model import strat_pred


# Construct a dataframe of articles, along with a list of persuasive strategies
# that they were annotated with.
def construct_article_df(directory):
  articles, strategies = [], []

  # Loop through the annotation folder
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # Open the json file with utf encoding
    with open(f, encoding="UTF-8") as read_file:
      file = json.load(read_file)

    # Extract the article text
    article = file['_referenced_fss']['12']['sofaString']
    articles.append(article)

    # Extract the annotations
    strategy_location = file['_views']['_InitialView']
    singular_strats = []

    # If this article was annotated with any strategies:
    if "Persuasive_Labels" in strategy_location:
      persuasive_labels = strategy_location["Persuasive_Labels"]
      for strat in persuasive_labels:
        # Extract the annotation name
        try:
          annotation = list(strat.items())[3][1]
          singular_strats.append(annotation)
        except:
          continue

    strategies.append(singular_strats)

  data = pd.DataFrame(list(zip(articles, strategies)), columns=["article", "target_strategy"])
  return data


# Remove duplicates from a list of lists
def remove_duplicates(list1):
  list2 = []
  for inner_list in list1:
    list2.append(list(set(inner_list)))
  return list2


# Convert a list of lists into a list of strings, separated by the given token
def list_to_str(list1, sep=' '):
  list2 = []
  for inner_list in list1:
    list2.append(sep.join(inner_list))
  return list2


# Given a string, return its length in tokens
def token_length(new_str):
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  tokens = tokenizer(new_str, padding='max_length', max_length=1024,
                     truncation=True, return_tensors='pt')

  # Token arrays are numpy ones arrays, so add up all numbers
  # in the list that do not equal one
  length = 0
  for i in tokens['input_ids'][0]:
    if i != 1:
      length += 1
  return length


# Prepare MultiFC files to be used
def prepare_multiFC(file):
  data = pd.read_csv(file, sep="\t")

  data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]

  data = data.reset_index(drop=True)

  return data


# Add labels to our dataframe by cross referencing the multiFC dataset
def add_labels(articles, data, norm):
  labels = []

  for i in articles:
    i = i.replace("\ufeff", "")

    # Match the claim ID to a row in MultiFC and extract the label
    row = data.loc[data["claimID"] == i[:10]]
    try:
      labels.append(row["label"].tolist()[0])
    except:
      labels.append("none")

  norm_labels = {
    'mostly true': 'true',
    'truth!': 'true',
    'true': 'true',
    'in-the-green': 'true',
    'mostly truth!': 'true',
    'disputed!': 'mixed',
    'mostly false': 'false',
    'mostly fiction!': 'false',
    'in-the-red': 'false',
    'fiction!': 'false',
    'false': 'false',
    'none': 'none'
  }
  if norm:
    labels = pd.Series(labels).apply(lambda x: norm_labels[x])
  return labels


# Cut irrelevant pre-text from the article texts
def cut_pre_text(articles):
  new_articles = []
  for article in articles:
    content = re.split("actual content", article, flags=re.IGNORECASE)
    if len(content) > 1:
      new_article = content[1][2:]
      new_articles.append(new_article)
    else:
      new_articles.append(article)
  return new_articles


# Summarize the article text so that it can be combined with the persuasive
# strategies and inputted into Roberta
def correct_length_inputs(articles, strategies):
  combined = []

  # For all articles and the strategies they are annotated with
  for article, strategy in zip(articles, strategies):

    # The maximum token length the article can be is 512 - (strategy token length)
    strat_token_len = token_length(strategy)
    max_article_len = 512 - strat_token_len

    combined_str = ""
    for i in range(10, 0, -1):
      try:
        summary = summarize(article, ratio=i / 10)
        token_len = token_length(summary)
        if token_len < max_article_len:
          combined_str = summary + ' </s> ' + strategy
          break
      except:
        combined_str = article + ' </s> ' + strategy
        break
    try:
      combined_str
    except:
      summary = summarize(article, ratio=0.1)
      combined_str = summary + ' </s> ' + strategy
    combined.append(combined_str)
    clear_output(wait=True)

  return combined


def split_article(article, context):
  # Using the nltk "punkt" corpus, tokenize the article into sentences
  sentences = tokenize.sent_tokenize(article)

  # A list of all segments in the current file
  segments = []

  # Loop through the sentences of the article, making segments along the way.
  #
  # A segment is the focus sentence, followed by a separating character,
  # followed by two sentences to the left of the focus sentence, followed by
  # another separating character, followed by the two sentences to the right
  for idx, sentence in enumerate(sentences):

    # If the left context is out of range, leave it blank
    try:
      left1 = sentences[idx - 2] if (idx - 2) >= 0 else ""
    except:
      left1 = ""
    try:
      left2 = sentences[idx - 1] if (idx - 1) >= 0 else ""
    except:
      left2 = ""
    # If the right context is out of range, leave it blank
    try:
      right1 = sentences[idx + 1]
    except:
      right1 = ""
    try:
      right2 = sentences[idx + 2]
    except:
      right2 = ""

    if context == "none":
      segment = sentence
    elif context == "low":
      segment = sentence + " </s> " + left2 + " </s> " + right1
    elif context == "high":
      segment = sentence + " </s> " + left1 + left2 + " </s> " + right1 + right2
    else:
      raise "Invalid Context given (must be 'none', 'low', or 'high')"
    segments.append(segment)
  return segments


l2 = {
  0: "Narrative with Details",
  1: "Using Anecdotes and Personal Experience as Evidence",
  2: "distrusting government or pharmaceutical companies",
  3: "politicizing health issues",
  4: "Highlighting Uncertainty and Risk",
  5: "Exploiting Science’s Limitations",
  6: "inappropriate use of scientific evidence",
  7: "rhetorical tricks",
  8: "biased reasoning to make a conclusion",
  9: "emotional appeals",
  10: "distinctive linguistic features",
  11: "Establishing Legitimacy",
}

l3 = {
  0: "Narrative with details_verified to be false",
  1: "Narrative with details_details verified to be true",
  2: "Narrative with details_details not verified",
  3: "financial motivation",
  4: "freedom of choice and agency",
  5: "ingroup vs. outgroup",
  6: "political figures or political argument",
  7: "religion or ideology",
  8: "Inappropriate Use of Scientific and Other Evidence - out of context_verified",
  9: "less robust evidence or outdated evidence_verified",
  10: "lack of citation for evidence",
  11: "exaggeration",
  12: "inappropriate analogy or false connection",
  13: "wrong cause-effect",
  14: "lack of evidence or use unverified and incomplete evidence to make a claim",
  15: "claims without evidence",
  16: "evidence does not support conclusion",
  17: "shifting hypothesis",
  18: "fear",
  19: "anger",
  20: "hope",
  21: "anxiety",
  22: "uppercase words",
  23: "linguistic intensifier (e.g., extreme words)",
  24: "title of article as clickbait",
  25: "bolded words or underline",
  26: "ellipses, exaggerated/excessive usage of punctuation marks",
  27: "citing seemingly credible source",
  28: "surface credibility markers",
  29: "Call to action"
}

l4 = {
  0: "citing source to establish legitimacy_source verified to be credible",
  1: "citing source to establish legitimacy_source verified to not be credible in this context",
  2: "citing source to establish legitimacy_source not verified",
  3: "citing source to establish legitimacy_source verified to be made up",
  4: "legitimate persuasive techniques: rhetorical question",
  5: "legitimate persuasive techniques: humor",
  6: "surface credibility markers - medical or scientific jargon",
  7: "surface credibility markers - words associated with nature or healthiness",
  8: "surface credibility markers - simply claiming authority or credibility"
}


# Predict the persuasive strategies in a list of articles
def predict_strategies(articles, models, context):
  pred = []
  i=0
  # Loop through the list of articles
  for article in articles:

    article_strats = []

    # Split the article into sequences
    text_sequences = split_article(article, context)

    # Loop through the sequences
    for seq in text_sequences:
      # Predict the persuasive strategies at the three layers
      # of classification
      l2_output = strat_pred(models[1], seq, context)
      l2_output = np.array(l2_output > 0.5, dtype=float)
      l2_idx = np.where(l2_output == 1)[0]
      l2_output = [l2[idx] for idx in l2_idx]

      l3_output = strat_pred(models[2], seq, context)
      l3_output = np.array(l3_output > 0.5, dtype=float)
      l3_idx = np.where(l3_output == 1)[0]
      l3_output = [l3[idx] for idx in l3_idx]

      l4_output = strat_pred(models[3], seq, context)
      l4_output = np.array(l4_output > 0.5, dtype=float)
      l4_idx = np.where(l4_output == 1)[0]
      l4_output = [l4[idx] for idx in l4_idx]

      # Concatenate teh strategies
      seq_strats = l2_output + l3_output + l4_output

      # Add the sequence strategies to the article strategies
      article_strats += seq_strats
    print(i)
    i+=1
    pred.append(article_strats)

  return pred


# Build a dataset from the annotated articles
def build_complete_dataset(file, models, context):
  # Construct the basic dataframe with article texts and ground truth strategies
  data = construct_article_df(file)

  # Remove duplicates and convert the strategies into a token separated list
  data["target_strategy"] = remove_duplicates(data["target_strategy"])
  data["target_strategy"] = list_to_str(data["target_strategy"], sep=' </s> ')

  # Predict the article strategies at the given context
  data["pred_strategy"] = predict_strategies(data["article"], models, context)
  data["pred_strategy"] = remove_duplicates(data["HC_pred_strategy"])
  data["pred_strategy"] = list_to_str(data["HC_pred_strategy"], sep=' </s> ')

  # Prepare the multiFC files for labelling
  file = "data/all.tsv"
  multiFC_data = prepare_multiFC(file)

  # Add the normalized labels
  data["label"] = add_labels(data["article"], multiFC_data, norm=True)
  data = data[data["label"] != "mixed"]
  data = data[data["label"] != "none"]

  # Cut the text that is not part of the article content
  data["article"] = cut_pre_text(data["article"])

  # Create the column with the article text and target strategies
  data["target_combined"] = correct_length_inputs(data["article"], data["target_strategy"])

  # Create the column with the article text and predicted strategies
  data["pred_strategy"] = correct_length_inputs(data["article"], data["pred_strategy"])

  return data


# Article dataset used in training and testing
class ArticleDataset(torch.utils.data.Dataset):

  def __init__(self, data, column):
    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    labels = \
      {
        "true": 0,
        "false": 1
      }
    self.labels = [labels[label] for label in data['label']]
    self.texts = [self.tokenizer(str(text),
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True,
                                 return_tensors="pt")
                  for text in data[column]]

  def classes(self):
    return self.labels

  def __len__(self):
    return len(self.labels)

  def get_batch_labels(self, idx):
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    batch_y = self.get_batch_labels(idx)
    return batch_texts, batch_y
