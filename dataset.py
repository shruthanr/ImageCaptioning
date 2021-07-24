import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

spacy_eng = spacy.load("en")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        freqs = {}
        i = 4

        for s in sentence_list:
            for w in self.tokenizer(s):
                if w not in freqs:
                    freqs[w] = 1
                else:
                    freqs[w] += 1

                if freqs[w] == self.freq_threshold:
                    self.stoi[w] = i
                    self.itos[i] = w
                    i += 1

    def get_indices(self, text):
        tokenized = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK"] for token in tokenized
        ]


class FlickrDataset(Dataset):

    def __init__(self, root_dir, captions_file, transforms=None, freq_threshold=5):
        self.root_dir = root_dir
        self.data = pd.read_csv(captions_file)
        self.transforms = transforms

        self.imgs = self.data["image"]
        self.captions = self.data["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        caption = self.captions[item]
        img_id = self.imgs[item]

        img = Image.open(
            os.path.join(self.root_dir, img_id)
        ).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        caption_idx = [self.vocab.stoi["<SOS>"]]
        caption_idx += self.vocab.get_indices(caption)
        caption_idx.append(self.vocab.stoi["<EOS>"])
        caption_idx = torch.tensor(caption_idx)

        return img, caption_idx

