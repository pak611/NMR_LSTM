import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # Add batch_first=True

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # Embedding the input and reshaping to [batch_size, 1, hidden_size]
        embedded = self.embedding(x).view(-1, 1, self.hidden_size)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target_length):
        batch_size = source.size(0)
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_length, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(source)

        # Initial input to the decoder (e.g., <SOS> tokens)
        input = torch.zeros(batch_size).long().to(self.device)  # Shape: [batch_size]

        for t in range(target_length):
            # Embedding and reshaping handled inside the decoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            input = output.argmax(1)  # Shape: [batch_size]

            return outputs
