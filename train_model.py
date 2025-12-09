import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# specifies nn.Module as parent class, nn.Module is the base class for neural networks in pytorch
# make changes for better performance specific to use case
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolution layers
        # params: input channel, output channel, kernel size, stride
        # first layer: grayscale image = 1 input channel
        # output = 32 channels, kernel size = 3, stride = 1 (idk chat gpt suggested these)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # second layer: 32 input = output of first layer, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # randdomly turns off 25 and 50% respectively of the neurons, helps with robustness and overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Linear is a fully connected layer. Each input is connected to every output. 
        # plugs the input vector into linear equation y = xW + b, W and b are params that get updated during training
        # params are input, output. input is 64 features * (12 * 12) image (after passing thru conv layers + max pooling)
        # output is 128 features
        self.fc1 = nn.Linear(9216, 128)
        # final output is 10 features (numbers in the range of 0-9)
        self.fc2 = nn.Linear(128, 10)

    # overrides parent class definition for a forward pass
    # specifies the flow of data during a forward pass (all of these were suggested by chat gpt)
    # conv1 -> relu -> conv2 -> relu -> max pool -> dropout -> flatten -> fc1 -> relu -> dropout -> fc2 -> log softmax
    # relu is used to prevent linearity, which would cause the CNN to be unable to learn useful mappings (basically just sets negative vals to 0)
    # max pool reduces the size of of image, keeps most important data by keeping highest value in a 2x2 grid (highest activation -> more relevant)
    # flatten converts the output into a single vector regardless of original dimensions
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # softmax takes raw scores and converts it into a probability distribution (all values sum to 1)
        # log_softmax logs this distribution (makes it easier for computer to process)
        output = F.log_softmax(x, dim=1)
        return output

def train():
    # was originally gonna use gpu on my windows but cpu seems to work just fine 
    device = "cpu"

    # hyperparameters
    batch_size = 64 # num of images to process at once
    epochs = 5 # num of times to loop thru whole dataset (i do not have this kind of compute)
    learning_rate = 1.0 # rate of change of params during training (W and b)
    # learning rate was initially 0.01, way too small

    # definition for how to transform the data from the dataset into something that the CNN can process
    transform = transforms.Compose([
        # transforms image to tensor (pretty much just a multidimensional array, it's what pytorch uses)
        transforms.ToTensor(),
        # normalizes the image using the mean and standard deviation of the pixel intensity of the dataset (helps with training)
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # shuffles dataset and splits it into individual batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initialize model, sends it to the training device
    model = Net().to(device)
    # the algorithm that updates weights of params (W and b)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    # training loop
    print("Starting training...")
    model.train() # enables "training mode", enables dropout (since you don't want dropout when actually testing/using model)
    
    # goes through dataset epochs times
    for epoch in range(1, epochs + 1):
        for batch_id, (data, target) in enumerate(train_loader):
            # moves images (data) and labels (target) to training device
            data, target = data.to(device), target.to(device)
            # resets the optimizer by clearing gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # loss function
            loss = F.nll_loss(output, target)
            # backpropagation, calculates each param's contribution toward the error
            loss.backward()
            # update weights by adjusting in the direction of minimizing error
            optimizer.step()

            # logging to see progress in real time
            if batch_id % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_id * len(data)}/{len(train_loader.dataset)} ({100. * batch_id / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # save model
    print("Saving model to model.pth...")
    # makes it portable to main.py for testing
    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
