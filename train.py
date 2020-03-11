from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import LabelAccuracyEvaluator,EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import os
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = '../local/bert_vi/bert4news.pytorch'
batch_size = 16
texts_reader = LabelSentenceReader('./data')
examples = texts_reader.get_examples('train.csv')
train_num_labels = texts_reader.get_num_labels()
logging.info([e.texts for e in examples[0:10]])
logging.info("Num_labels : "+str(train_num_labels))

model_save_path = 'output/training_sa'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
try:
    folder_list = os.listdir("output/")
except:
    folder_list = []

if(len(folder_list) > 0):
    folder_list.sort()
    init_path = folder_list[-1]
    logging.info("Restore model from {}".format("output/"+init_path))
    model = SentenceTransformer("output/"+init_path)
else:
    logging.info("Create new model...")
    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.BERT(model_name,max_seq_length=200,do_lower_case=False)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=False,
                                pooling_mode_cls_token=True,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
logging.info("Read train dataset")
train_data = SentencesDataset(texts_reader.get_examples('train.csv'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.Softmax1Loss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read dev dataset")
dev_data = SentencesDataset(examples=texts_reader.get_examples('dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = LabelAccuracyEvaluator(dev_dataloader,softmax_model=train_loss)
logging.info(texts_reader.get_num_labels())

# Configure the training
num_epochs = 20

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=400,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )

# predict model

