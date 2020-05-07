# BERT for Vietnamese on more 20 GB news dataset

Apply for task sentiment analysis on using [AIViVN's comments dataset](https://www.aivivn.com/contests/6)

***************New Mar 11 , 2020 ***************

**[BERT](https://github.com/google-research/bert)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

You can download trained model:
- [tensorflow](https://drive.google.com/file/d/1X-sRDYf7moS_h61J3L79NkMVGHP-P-k5/view?usp=sharing).
- [pytorch](https://drive.google.com/file/d/11aFSTpYIurn-oI2XpAmcCTccB_AonMOu/view?usp=sharing).



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