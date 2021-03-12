# BERT for Vietnamese is trained on more 20 GB news dataset

Apply for task sentiment analysis on using [AIViVN's comments dataset](https://www.aivivn.com/contests/6)

The model achieved 0.90268 on the public leaderboard, (winner's score is 0.90087)
Bert4news is used for a toolkit Vietnames(segmentation and Named Entity Recognition) at ViNLPtoolkit(https://github.com/bino282/ViNLP)

***************New Mar 11 , 2020 ***************

**[BERT](https://github.com/google-research/bert)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

We use word sentencepiece, use basic bert tokenization and same config with bert base with lowercase = False.

You can download trained model:
- [tensorflow](https://drive.google.com/drive/folders/1d_MFVi32YRZTBHDNQahAGyqlzetS8XLU?usp=sharing).
- [pytorch](https://drive.google.com/drive/folders/1-vAaQTdaLwMk2rEyTXOsTZAfZ7MTlyIQ?usp=sharing).

Use with huggingface/transformers
``` bash
import torch
from transformers import AutoTokenizer,AutoModel
tokenizer= AutoTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
bert_model = AutoModel.from_pretrained("NlpHUST/vibert4news-base-cased")

line = "Tôi là sinh viên trường Bách Khoa Hà Nội ."
input_id = tokenizer.encode(line,add_special_tokens = True)
att_mask = [int(token_id > 0) for token_id in input_id]
input_ids = torch.tensor([input_id])
att_masks = torch.tensor([att_mask])
with torch.no_grad():
    features = bert_model(input_ids,att_masks)

print(features)

```

Run training with base config

``` bash

python train_pytorch.py \
  --model_path=bert4news.pytorch \
  --max_len=200 \
  --batch_size=16 \
  --epochs=6 \
  --lr=2e-5

```

### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
