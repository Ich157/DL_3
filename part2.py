import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv layer with relu + pool
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten
        x = self.fc1(x)
        return x


policy = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor()
                        ])

#hyperparameters
#batch = [4,8,16,32,64]
#learning_rates = [0.0001,0.001,0.01,0.1]
#epoches = [2,5,10,20]
batch = [4]
learning_rates = [0.001]
epoches = [5]

# load and split data into train and val set
#train = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=policy)
train = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transforms.ToTensor())
train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])
test = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())

net = Net()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


## Training of the network
best_parameters =[0,0,0]
best_acc = 0
for n_epochs in epoches:
    for batch_size in batch:
        for lr in learning_rates:
            print("parameters:")
            print(f"epoches: {n_epochs},batchSize: {batch_size},lr: {lr}")
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)

            net = Net()
            net.load_state_dict(torch.load(PATH))

            # loss
            criterion = nn.CrossEntropyLoss()
            # optimizer
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            train_loss = []
            val_losses = []
            for epoch in range(n_epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                            # get the inputs; data is a list of [inputs, labels]
                            inputs, labels = data

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                            # print statistics
                            running_loss += loss.item()
                print(f"train loss after {epoch+1} epochs: {running_loss/len(train_set)}")
                train_loss.append(running_loss/len(train_set))
                # Validation loss
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                for i, data in enumerate(valloader, 0):
                        inputs, labels = data

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                print(f"val loss after {epoch+1} epochs: {val_loss/len(val_set)}")
                val_losses.append(val_loss/len(val_set))

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = net(images)
                        # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()



            accuracy = 100 * correct // total
            if accuracy > best_acc:
                best_acc = accuracy
                best_parameters = [n_epochs,batch_size,lr]
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # Here we save a checkpoint. It is automatically registered with
    # Ray Tune and can be accessed through `session.get_checkpoint()`
    # API in future iterations.

print(best_acc)
print(best_parameters)
print(train_loss)
print(val_losses)

plt.plot(train_loss, label = "Training loss")
plt.plot(val_losses, label = "Validation loss")
plt.legend()
plt.title("Average loss over epoches")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.show()
