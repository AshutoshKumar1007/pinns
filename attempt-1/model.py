import torch


class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        width = 100  # Make the network wider

        # Define all layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1)  # Final layer maps to a single output
        )

    def forward(self, x, y):
        # Using nn.Sequential makes the forward pass clean and error-proof
        inputs = torch.cat([x, y], axis=1)
        return self.layers(inputs)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)