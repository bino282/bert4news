from . import InputExample
import csv
import gzip
import os
import pandas as pd
class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""
    def __init__(self, folder):
        self.folder = folder
        self.label_map = {0:0,1:1}

    def get_examples(self, filename, max_examples=0):
        examples = []

        id = 0
        df = pd.read_csv(os.path.join(self.folder, filename))
        for index, row in df.iterrows():
            label = row["label"]
            sentence = row["text"]

            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

            label_id = self.label_map[label]
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence], label=label_id))

            if 0 < max_examples <= id:
                break

        return examples

    def get_labels(self):
        return self.label_map

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]