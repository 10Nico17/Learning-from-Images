# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

CATEGORIES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


# nur für um Datensatz anzuschauen
def show_random_images_labels(loader, num_images=10):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    # Zufällige Auswahl von 10 Indizes
    random_indices = np.random.choice(len(images), num_images, replace=False)    
    # Extrahieren der zufälligen Bilder und Labels
    random_images = images[random_indices]
    random_labels = labels[random_indices]
    # Anzeigen der Shapes der zufälligen Bilder
    for i in range(num_images):
        print(f"Shape of Image {i + 1}: {random_images[i].shape}")
    # Anzeigen der zufälligen Bilder
    grid_img = utils.make_grid(random_images, nrow=num_images, padding=10)
    plt.imshow(np.transpose(grid_img, (1, 2, 0)))
    plt.title('Random Images with Labels')
    plt.axis('off')
    plt.show()


# implement your own NNs
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # TODO: YOUR CODE HERE
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)        
        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)       
        # Dropout Layer
        self.dropout = nn.Dropout(p=0.25)         
        # Fully Connected Layers
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 10)  

    def forward(self, x):
        # TODO: YOUR CODE HERE     
        #print('step0 x shape: ', x.shape)       
        x = F.relu(self.conv1(x))
        #print('step1 x shape: ', x.shape)       
        x = F.relu(self.conv2(x))        
        #print('step2 x shape: ', x.shape)    
        x = self.pool(x)        
        #print('step3 x shape: ', x.shape)  
        x = self.dropout(x)
        #print('step4 x shape: ', x.shape) 
        x = F.relu(self.conv3(x))
        #print('step5 x shape: ', x.shape)       
        x = F.relu(self.conv4(x))        
        #print('step6 x shape: ', x.shape)       
        x = self.pool(x)
        #print('step7 x shape: ', x.shape)       
        x = self.dropout(x)
        #print('step8 x shape: ', x.shape)       
        x = torch.flatten(x, start_dim=1)
        #print('x after flatten: ', x.shape)
        # Fully Connected Layers with ReLU activation and Dropout
        x = F.relu(self.fc1(x))
        #print('x after fc1: ', x.shape)
        x = self.fc2(x)     
        #print('x after fc2: ', x.shape)
        return x

    def name(self):
        return "MyNeuralNetwork"

# trained on BHT cluster
# network structure from AlexNet
# source: https://blog.paperspace.com/alexnet-pytorch/

class MyNeuralNetwork2(nn.Module):  
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2304, 1024),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(512, 10))
    def forward(self, x):
        #print('step0 x shape: ', x.shape)      
        out = self.layer1(x)
        #print('step1 x shape: ', out.shape)
        out = self.layer2(out)
        #print('step2 x shape: ', out.shape)
        out = self.layer3(out)
        #print('step3 x shape: ', out.shape)      
        out = self.layer4(out)
        #print('step4 x shape: ', out.shape)      
        out = self.layer5(out)
        #print('step5 x shape: ', out.shape)      
        out = out.reshape(out.size(0), -1)
        #print('step6 x shape: ', out.shape)      
        out = self.fc(out)
        #print('step7 x shape: ', out.shape)      
        out = self.fc1(out)
        #print('step8 x shape: ', out.shape)      
        out = self.fc2(out)
        #print('step9 x shape: ', out.shape)  
        return out
    
    def name(self):
        return "MyNeuralNetwork2"



# similar to number1 but more parameters
class MyNeuralNetwork3(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork3, self).__init__()
        # TODO: YOUR CODE HERE
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 86, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(86, 56, kernel_size=3, stride=2, padding=1) 
        self.conv5 = nn.Conv2d(56, 32, kernel_size=3, stride=2, padding=1) 
        # MaxPooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)       
        # Dropout Layer
        self.dropout1 = nn.Dropout(p=0.5) 
        self.dropout2 = nn.Dropout(p=0.25) 
        # Fully Connected Layers
        self.fc1 = nn.Linear(9464, 512)
        self.fc2 = nn.Linear(512, 10)  

    def forward(self, x):
        # TODO: YOUR CODE HERE     
        #print('step0 x shape: ', x.shape)       
        x = F.relu(self.conv1(x))
        #print('step1 x shape: ', x.shape)       
        x = F.relu(self.conv2(x))        
        #print('step2 x shape: ', x.shape)    
        x = self.pool(x)        
        #print('step3 x shape: ', x.shape)  
        x = self.dropout1(x)
        #print('step4 x shape: ', x.shape) 
        x = F.relu(self.conv3(x))
        #print('step5 x shape: ', x.shape)       
        x = F.relu(self.conv4(x))             
        #print('step6 x shape: ', x.shape)       
        x = self.pool(x)
        #print('step7 x shape: ', x.shape)       
        x = self.dropout2(x)
        #print('step8 x shape: ', x.shape)       
        x = torch.flatten(x, start_dim=1)
        #print('x after flatten: ', x.shape)
        # Fully Connected Layers with ReLU activation and Dropout
        x = F.relu(self.fc1(x))
        #print('x after fc1: ', x.shape)
        x = self.fc2(x)     
        #print('x after fc2: ', x.shape)
        return x

    def name(self):
        return "MyNeuralNetwork3"
    
    
    
def training(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        # zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        # backward
        loss.backward()
        optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
    return epoch_loss, epoch_acc


def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc



def plot(train_history, test_history, metric, num_epochs):
    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}.png")
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(0)

# hyperparameter
# TODO: find good hyperparameters
batch_size = 64
num_epochs = 6
learning_rate = 0.005
momentum = 0.8



train_transform = transforms.Compose([
    #transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),  # Beispiel für eine zusätzliche Transformation
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    #transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# load train and test data
root = './data'
train_set = datasets.FashionMNIST(root=root,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
test_set = datasets.FashionMNIST(root=root,
                                 train=False,
                                 transform=test_transform,
                                 download=True)

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading
}

train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)


#show_random_images_labels(train_loader, num_images=10)


## model setup
model = MyNeuralNetwork().to(device)
#model = MyNeuralNetwork2().to(device)
#model = MyNeuralNetwork3().to(device)
print(model)



optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()


train_acc_history = []
test_acc_history = []

train_loss_history = []
test_loss_history = []

best_acc = 0.0
since = time.time()

for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    # train
    training_loss, training_acc = training(model, train_loader, optimizer,
                                           criterion, device)
    train_loss_history.append(training_loss)
    train_acc_history.append(training_acc)
    # test
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    
    ## overall best model
    if test_acc > best_acc:
        best_acc = test_acc
        #  best_model_wts = copy.deepcopy(model.state_dict())


time_elapsed = time.time() - since
print(
    f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
)
print(f'Best val Acc: {best_acc:4f}')


# plot loss and accuracy curves
train_acc_history = [h.cpu().numpy() for h in train_acc_history]
test_acc_history = [h.cpu().numpy() for h in test_acc_history]
plot(train_acc_history, test_acc_history, 'accuracy', num_epochs)
plot(train_loss_history, test_loss_history, 'loss', num_epochs)


'''
# plot examples
example_data, _ = next(iter(test_loader))
with torch.no_grad():
    output = model(example_data)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Pred: {}".format(CATEGORIES[output.data.max(
            1, keepdim=True)[1][i].item()]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("examples.png")
    plt.show()
'''