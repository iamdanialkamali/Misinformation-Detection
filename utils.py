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

def get_collate_fn():
    from transformers import LongformerTokenizer
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    def collate_fn(batch):
        """
        data: is a list of tuples with (claim, label)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        texts,labels = torch.utils.data.default_collate(batch)
        texts = tokenizer(texts,
                            padding=True,
                            # max_length=4096,
                            truncation=True,
                            return_tensors="pt")
        return texts, labels
    return collate_fn