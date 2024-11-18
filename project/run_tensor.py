"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import math


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


# TODO: Implement for Task 2.5.
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

        # xavier_scale = (2.0 / (in_size + out_size)) ** 0.5
        # self.weights = minitorch.Parameter(
        #     xavier_scale * (2 * minitorch.rand((in_size, out_size)) - 1)
        # )
        # # Initialize bias to zeros
        # self.bias = minitorch.Parameter(minitorch.zeros((out_size,)))

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        # x shape: (batch_size, in_size)
        # weights shape: (in_size, out_size)
        batch_size, in_size = x.shape[0], x.shape[1]
        out_size = self.weights.value.shape[1]

        # reshape x from (batch_size, in_size) to (batch_size, in_size, 1)
        x_3d = x.view(batch_size, in_size, 1)

        # reshape weights from (in_size, out_size) to (1, in_size, out_size)
        w_3d = self.weights.value.view(1, self.weights.value.shape[0], self.weights.value.shape[1])

        # This will broadcast to (batch_size, in_size, out_size)
        # Then sum along dimension 1 (in_size), then reduce to get (batch_size, out_size)
        out = (x_3d*w_3d).sum(1).view(batch_size, out_size)

        return out + self.bias.value


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
