from torch import nn

class BracketPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_layers=[32, 64]):
        super().__init__()

        layers = []
        in_features = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # Output layer (binary classification → 1 neuron)
        layers.append(nn.Linear(in_features, 1))

        self.linear = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.linear(x)
