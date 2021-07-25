import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as torch_transforms

spacy_eng = spacy.load("en_core_web_sm")


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
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized
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


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(root_folder, captions_file, transforms, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):

    dataset = FlickrDataset(root_folder, captions_file, transforms=transforms)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx)
    )

    return loader


def main():
    root_dir = "../../Datasets/flickr8k/images"
    captions_file = "../../Datasets/flickr8k/captions.txt"
    transforms = torch_transforms.Compose(
        [
            torch_transforms.Resize((224, 224)),
            torch_transforms.ToTensor()
        ]
    )
    data_loader = get_loader(root_dir, captions_file, transforms=transforms)

    for idx, (imgs, captions) in enumerate(data_loader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()
