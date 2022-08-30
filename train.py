import torch
from torch import nn
from model import custom_model, CustomConvNet

# Training
hyper_param_epoch = 10
hyper_param_batch = 8
hyper_param_learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 10
model = CustomConvNet(num_classes=num_classes).to(device)

def train(model, train_loader, optimizer, hyper_param_epoch):
    model.train()
    for e in range(hyper_param_epoch):
        for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                    .format(e + 1, hyper_param_epoch, loss.item()))