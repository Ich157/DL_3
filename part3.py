import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt

class Net(nn.Module):
    N = 80

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, self.N, 3, stride=1, padding=1)
        #self.global_pooling = nn.AdaptiveAvgPool2d([1, 1])
        self.global_pooling = nn.AdaptiveMaxPool2d([1,1])
        self.fc1 = nn.Linear(self.N, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv layer with relu + pool
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pooling(x)
        x = torch.flatten(x, 1) # global max pool
        x = self.fc1(x)
        return x

def split_into_shapes(set):
    set_32 = []
    set_48 = []
    set_64 = []

    for image, label in set:
        if image.shape[1] == 32:
            set_32.append((image,label))
        if image.shape[1] == 48:
            set_48.append((image, label))
        if image.shape[1] == 64:
            set_64.append((image, label))

    return set_32,set_48,set_64

net = Net()

PATH = './part3.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

print("Summary")
print(summary(net, (3,16,16)))

train_images = torchvision.datasets.ImageFolder(root="mnist-varres/train", transform=transforms.ToTensor())
test_images = torchvision.datasets.ImageFolder(root="mnist-varres/test", transform=transforms.ToTensor())

train_set, val_set = torch.utils.data.random_split(train_images, [50000, 10000])

train_set_32, train_set_48, train_set_64 = split_into_shapes(train_set)
val_set_32, val_set_48, val_set_64 = split_into_shapes(val_set)
test_set_32, test_set_48, test_set_64 = split_into_shapes(test_images)



# hyperparameters
batch = [4]
learning_rates = [0.001]
epoches = [8]

best_parameters = [0, 0, 0]
best_acc = 0

for batch_size in batch:
    for lr in learning_rates:
        for epoch in epoches:

            net = Net()
            net.load_state_dict(torch.load(PATH))

            print("parameters:")
            print(f"epoches: {epoch},batchSize: {batch_size},lr: {lr}")
            train_loss = [0]
            val_losses = [0]

            # loss
            criterion = nn.CrossEntropyLoss()
            # optimizer
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            trainloader_32 = torch.utils.data.DataLoader(train_set_32, batch_size=batch_size, shuffle=True, num_workers=2)
            trainloader_48 = torch.utils.data.DataLoader(train_set_48, batch_size=batch_size, shuffle=True, num_workers=2)
            trainloader_64 = torch.utils.data.DataLoader(train_set_64, batch_size=batch_size, shuffle=True, num_workers=2)

            trainloader = [trainloader_32,trainloader_48,trainloader_64]

            valloader_32 = torch.utils.data.DataLoader(val_set_32, batch_size=batch_size, shuffle=True, num_workers=2)
            valloader_48 = torch.utils.data.DataLoader(val_set_48, batch_size=batch_size, shuffle=True, num_workers=2)
            valloader_64 = torch.utils.data.DataLoader(val_set_64, batch_size=batch_size, shuffle=True, num_workers=2)

            valloader = [valloader_32,valloader_48,valloader_64]

            testloader_32 = torch.utils.data.DataLoader(test_set_32, batch_size=batch_size, shuffle=True, num_workers=2)
            testloader_48 = torch.utils.data.DataLoader(test_set_48, batch_size=batch_size, shuffle=True, num_workers=2)
            testloader_64 = torch.utils.data.DataLoader(test_set_64, batch_size=batch_size, shuffle=True, num_workers=2)

            testloader = [testloader_32, testloader_48, testloader_64]

            for epoch in range(epoch):  # loop over the dataset multiple times

                running_loss = 0.0
                for loader in trainloader:
                    for i, data in enumerate(loader, 0):
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
                print(f"train loss after {epoch + 1} epochs: {running_loss / len(train_set)}")
                train_loss.append(running_loss / len(train_set))
                 # Validation loss
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                for loader in valloader:
                    for i, data in enumerate(loader, 0):
                        inputs, labels = data

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                print(f"val loss after {epoch + 1} epochs: {val_loss / len(val_set)}")
                val_losses.append(val_loss / len(val_set))

            correct = 0
            total = 0
            with torch.no_grad():
                for loader in testloader:
                    for data in loader:
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
                best_parameters = [epoch, batch_size, lr]
                best_train_loss = train_loss
                best_val_losses = val_losses
            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
# Here we save a checkpoint. It is automatically registered with
# Ray Tune and can be accessed through `session.get_checkpoint()`
# API in future iterations.

print(best_acc)
print(best_parameters)


plt.plot(best_train_loss, label = "Training loss")
plt.plot(best_val_losses, label = "Validation loss")
plt.legend()
plt.title("Average loss over epoches")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.xlim(1,8)
plt.show()