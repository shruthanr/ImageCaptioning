import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self, embed_dim, train_cnn=False):
        super(Encoder, self).__init__()
        self.train_cnn = train_cnn
        self.inception_model = models.inception_v3(pretrained=True)
        self.inception_model.aux_logits = False
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception_model(images)

        # # Train only the final fully-conncted layer
        # for name, param in self.inception_model.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_cnn
        return self.dropout(self.relu(features))


class Decoder(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_len, n_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_len, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, vocab_len)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        x = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.lstm(x)
        out = self.linear(hidden)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_len, n_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim, hidden_dim, vocab_len, n_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        out = self.decoder(features, captions)
        return out

    def caption_image(self, image, vocab, max_len=50):
        res_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            lstm_states = None

            for _ in range(max_len):
                h, c = self.decoder.lstm(x, lstm_states)
                out = self.decoder.linear(h.squeeze(0))
                prediction = out.argmax(1)

                res_caption.append(prediction.item())
                x = self.decoder.embed(prediction).unsqueeze(0)

                if vocab.itos[prediction.item()] == "<EOS>":
                    break

        return [vocab.itos[i] for i in res_caption]
