from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import torch



def calculate_sentence_transformer_embedding(text_to_encode):
    num = len(text_to_encode)
    emb_model = SentenceTransformer('/data/home/llm_dev/gaohongfu/flan')
    embeddings = []
    bar = tqdm(range(0, num, 20),desc='calculate embeddings')
    for i in range(0, num, 20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings