from modules import *


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_rnn_cells=1) -> None:
        super(RNN, self).__init__()
        self.register_buffer('vocab_size', torch.tensor(vocab_size))
        self.register_buffer('embedding_dim', torch.tensor(embedding_dim))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim))
        self.register_buffer('output_dim', torch.tensor(output_dim))
        self.register_buffer('num_rnn_cells', torch.tensor(num_rnn_cells))

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.rnn_cells = nn.ModuleList()
        for rnn_cell_idx in range(self.num_rnn_cells):
            if rnn_cell_idx == 0:
                self.rnn_cells.append(RNNCell(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim))
            else:
                self.rnn_cells.append(RNNCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim))

        self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        batch_size, sequence_length, embedding_dim = x.size()
        hidden_states = [torch.zeros(size=(batch_size, self.hidden_dim), device=x.device) for rnn_cell_idx in range(self.num_rnn_cells)]
        outputs = []

        for time in range(sequence_length):
            next_input = x[:, time, :]
            for rnn_cell_idx in range(self.num_rnn_cells):
                hidden_states[rnn_cell_idx] = self.rnn_cells[rnn_cell_idx](next_input, hidden_states[rnn_cell_idx])
                next_input = hidden_states[rnn_cell_idx]
            outputs.append(self.output_layer(hidden_states[-1]))

        outputs = torch.stack(outputs).permute(dims=(1, 0, 2))
        return outputs


class LSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_lstm_cells=1) -> None:
        super(LSTM, self).__init__()
        self.register_buffer('vocab_size', torch.tensor(vocab_size))
        self.register_buffer('embedding_dim', torch.tensor(embedding_dim))
        self.register_buffer('hidden_dim', torch.tensor(hidden_dim))
        self.register_buffer('output_dim', torch.tensor(output_dim))
        self.register_buffer('num_lstm_cells', torch.tensor(num_lstm_cells))

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.lstm_cells = nn.ModuleList()
        for lstm_cell_idx in range(self.num_lstm_cells):
            if lstm_cell_idx == 0:
                self.lstm_cells.append(LSTMCell(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim))
            else:
                self.lstm_cells.append(LSTMCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim))

        self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        batch_size, sequence_length, embedding_dim = x.size()
        hidden_states = [torch.zeros(size=(batch_size, self.hidden_dim), device=x.device) for lstm_cell_idx in range(self.num_lstm_cells)]
        contexts = [torch.zeros(size=(batch_size, self.hidden_dim), device=x.device) for lstm_cell_idx in range(self.num_lstm_cells)]
        outputs = []

        for time in range(sequence_length):
            next_input = x[:, time, :]
            for lstm_cell_idx in range(self.num_lstm_cells):
                hidden_states[lstm_cell_idx], contexts[lstm_cell_idx] = self.lstm_cells[lstm_cell_idx](
                    next_input, hidden_states[lstm_cell_idx], contexts[lstm_cell_idx]
                )
                next_input = hidden_states[lstm_cell_idx]
            outputs.append(self.output_layer(hidden_states[-1]))

        outputs = torch.stack(outputs).permute(dims=(1, 0, 2))
        return outputs
