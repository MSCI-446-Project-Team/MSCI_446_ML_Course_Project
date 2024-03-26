import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_features, n_hidden=51, output_features=1, dropout_rate=0.2):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(input_features, n_hidden)
        self.lstm2 = nn.LSTMCell(n_hidden, n_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        # self.linear = nn.Linear(n_hidden, output_features)
        self.linear = nn.Sequential(
            nn.Linear(n_hidden, output_features),
            nn.ReLU()
        )

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)
        h_t = torch.zeros(n_samples, self.n_hidden,
                          dtype=torch.float32, device=x.device)
        c_t = torch.zeros(n_samples, self.n_hidden,
                          dtype=torch.float32, device=x.device)
        h_t2 = torch.zeros(n_samples, self.n_hidden,
                           dtype=torch.float32, device=x.device)
        c_t2 = torch.zeros(n_samples, self.n_hidden,
                           dtype=torch.float32, device=x.device)

        for input_t in x.chunk(x.size(1), dim=1):
            input_t = input_t.squeeze(1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2 = self.dropout(h_t2)
            output = self.linear(h_t2)
            outputs.append(output)

        last_output = output

        for i in range(future):
            last_output = output.repeat(1, 5)

            last_output = last_output.squeeze(-1)

            h_t, c_t = self.lstm1(last_output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            last_output = output

        outputs = torch.cat(outputs, dim=1)

        return outputs
