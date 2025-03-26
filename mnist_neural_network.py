# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
    
        lower_bound = -1 / torch.sqrt(torch.tensor(d, dtype=torch.float32))
        upper_bound = 1 / torch.sqrt(torch.tensor(d, dtype=torch.float32))
        u = Uniform(lower_bound, upper_bound)

        self.w0 = Parameter(u.sample((h, d)))
        self.w1 = Parameter(u.sample((k, h)))
        self.b0 = Parameter(u.sample((h,)))  
        self.b1 = Parameter(u.sample((k,)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(x.float() @ self.w0.T + self.b0) @ self.w1.T + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        u = Uniform(-1 / math.sqrt(d), 1 / math.sqrt(d))

        self.w0 = Parameter(u.sample((h0, d)))
        self.w1 = Parameter(u.sample((h1, h0)))
        self.w2 = Parameter(u.sample((k, h1)))
        self.b0 = Parameter(u.sample((h0,))) 
        self.b1 = Parameter(u.sample((h1,))) 
        self.b2 = Parameter(u.sample((k,)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(relu(x.float() @ self.w0.T + self.b0) @ self.w1.T + self.b1) @ self.w2.T + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    accuracy = 0
    losses = []
    
    model.train()

    while accuracy < 0.99:

        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:

            pred = model(x)
            loss = cross_entropy(pred, y)
            
            pred_labels = torch.argmax(pred, dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        losses.append(avg_loss)
        print(f"accuracy: {accuracy}")

        if accuracy >= 0.99:
            break

    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    f1_model = F1(64, 784, 10)
    f2_model = F2(32, 32, 784, 10)
    dataset_train = TensorDataset(x, y)
    print("training f1")
    f1_train_loss = train(f1_model, Adam(f1_model.parameters()), DataLoader(dataset_train, shuffle=True, batch_size=128))
    print("training f2")
    f2_train_loss = train(f2_model, Adam(f2_model.parameters(), lr=0.001), DataLoader(dataset_train, shuffle=True, batch_size=128))
    plt.figure(figsize=(12, 8))
    
    plt.plot(range(0, len(f1_train_loss)), f1_train_loss, label="F1")
    plt.plot(range(0, len(f2_train_loss)), f2_train_loss, label="F2")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()

    plt.grid(True)
    plt.show()


    f1_model.eval()
    f2_model.eval()
    
    with torch.no_grad():
        f1_pred = f1_model(x_test)
        f2_pred = f2_model(x_test)
        
        f1_loss = cross_entropy(f1_pred, y_test).item()
        f2_loss = cross_entropy(f2_pred, y_test).item()
        
        f1_accuracy = (torch.argmax(f1_pred, dim=1) == y_test).float().mean().item()
        f2_accuracy = (torch.argmax(f2_pred, dim=1) == y_test).float().mean().item()
    
    print(f"F1: cross entropy loss = {f1_loss},  accuracy = {f1_accuracy}")
    print(f"F2: cross entropy loss = {f2_loss},  accuracy = {f2_accuracy}")

    f1_params = sum(p.numel() for p in f1_model.parameters())
    f2_params = sum(p.numel() for p in f2_model.parameters())
    print(f"F1: num of parameters = {f1_params}")
    print(f"F2: num of parameters = {f2_params}")




if __name__ == "__main__":
    main()
