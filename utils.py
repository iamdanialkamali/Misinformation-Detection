import torch
import numpy as np
import random 
import os
def set_seed(SEED):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_collate_fn(longformer=False):
    tokenizer = get_tokenizer(longformer)
    def collate_fn(batch):
        """
        data: is a list of tuples with (claim, label)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        trans = lambda text: text if not isinstance(text,float) else ""
        batch = [(trans(item[0]),item[1]) for item in batch]
        texts,labels = torch.utils.data.default_collate(batch)
        texts = tokenizer(texts,
                            padding=True,
                            max_length=4096 if longformer else 512,
                            truncation=True,
                            return_tensors="pt")
        return texts, labels
    return collate_fn


def get_tokenizer(longformer=False):
    from transformers import LongformerTokenizer, RobertaTokenizer
    if longformer:
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return tokenizer


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


def update_tokenizer(tokenizer):
    return tokenizer
    tokens = []
    for layer in [l2,l3,l4]:
        for value in layer.values():
            token = value.replace(" ","_").lower()
            tokens.append(token)
    tokenizer.add_tokens(tokens)
    return tokenizer