import torch
import torchvision
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
batch_size = 100
learning_rate = 0.001
num_classes = 10
transform_train = torchvision.transforms.Compose([torchvision.transforms.Pad(4),
                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.RandomCrop(32),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', download=False, train=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', download=False, train=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
cfg = {'VGG-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
class VGGNet(torch.nn.Module):
    def __init__(self, VGG_type, num_classes):
        super(VGGNet, self).__init__()
        self.features = self._make_layers(cfg[VGG_type])
        self.classifier = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M': 
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           torch.nn.BatchNorm2d(x),
                           torch.nn.ReLU(inplace=True)]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)
net_name = 'VGG-16'
model = VGGNet(net_name, num_classes).to(device)
summary(model,input_size=(3,32,32))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
import gc
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    gc.collect()
    torch.cuda.empty_cache()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
torch.save(model.state_dict(), 'VGG-16.ckpt')



