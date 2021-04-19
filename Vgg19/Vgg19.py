import parameters
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
device=parameters.dev

# Data load and prepare
transform = transforms.Compose([transforms.RandomHorizontalFlip(),        # In this case i didn't use many transforms I keep 2 because they perform so well in this model 
                                transforms.RandomRotation(0.2),
                                transforms.ToTensor(),
                                transforms.Resize((parameters.IMG_SIZE,parameters.IMG_SIZE))
                               ])

dataset = torchvision.datasets.ImageFolder(root = parameters.data_path,
                                           transform = transform)
print("No of Classes: ", len(dataset.classes))

train, val = torch.utils.data.random_split(dataset, [70, 20])   # those values depend on the train data I put 70 and 20 because i got 90 train image 

train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = parameters.batch_size,
                                           shuffle = False)

val_loader = torch.utils.data.DataLoader(dataset = val,
                                         batch_size = parameters.batch_size,
                                         shuffle = False)

# end

# import VGG model
vgg = torchvision.models.vgg19(pretrained=True)

vgg.classifier[6].out_features = 2
for param in vgg.features.parameters():
    param.requires_grad = False

vgg = vgg.cpu()

criterion = parameters.criteria
optimizer = torch.optim.Adam(vgg.parameters(), lr=parameters.learning_rate)
# end import and param

## training:
n=parameters.epochs
total_step = len(train_loader)
Loss = []
Acc = []
Val_Loss = []
Val_Acc = []

for epoch in range(n):
    acc = 0
    val_acc = 0
    for i, (images, labels) in enumerate(train_loader):
        vgg.train()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = vgg(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Checking accuracy
        preds = outputs.data.max(dim=1, keepdim=True)[1]
        acc += preds.eq(labels.data.view_as(preds)).cpu().sum()

    acc = acc / len(train_loader.dataset) * 100

    for i, (images, labels) in enumerate(val_loader):
        vgg.eval()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = vgg(images)
        val_loss = criterion(outputs, labels)

        # Checking accuracy
        preds = outputs.data.max(dim=1, keepdim=True)[1]
        val_acc += preds.eq(labels.data.view_as(preds)).cpu().sum()

    val_acc = val_acc / len(val_loader.dataset) * 100

    print(
    "Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   Val_loss : {val_loss:.2f};   Val_Accuracy : {val_acc:.2f}%".format(
        epoch + 1, loss=loss.item(), acc=acc, val_loss=val_loss.item(), val_acc=val_acc))

    Loss.append(loss)
    Acc.append(acc)

    Val_Loss.append(val_loss)
    Val_Acc.append(val_acc)
# end training

# 1st plot :
plt.plot(range(n),Loss)
plt.plot(range(n),Val_Loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()
# end 1st plot

# 2nd Plot
plt.plot(range(n),Acc)
plt.plot(range(n),Val_Acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()
# end 2nd plot

# model Evaluation
vgg.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        y_pred = []
        Images = images.to(device)
        Labels = labels.to(device)
        outputs = vgg(Images)
        prediction_array = outputs.data

        _, predicted = torch.max(prediction_array, 1)
        y_pred += predicted
        total += Labels.size(0)
        correct += (predicted == Labels).sum().item()

    acc = 100 * correct / total
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
# end evaluation

# model Evaluation on another set


testset = torchvision.datasets.ImageFolder(root = parameters.test_path,
                                           transform = transform)
print("No of Classes: ", len(testset.classes))


test_loader = torch.utils.data.DataLoader(dataset = testset,
                                           batch_size = parameters.batch_size,
                                           shuffle = False)
vgg.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        y_pred = []
        Images = images.to(device)
        Labels = labels.to(device)
        outputs = vgg(Images)
        prediction_array = outputs.data

        _, predicted = torch.max(prediction_array, 1)
        y_pred += predicted
        total += Labels.size(0)
        correct += (predicted == Labels).sum().item()

    acc = 100 * correct / total
    print('Test Accuracy of the model on the unknown images: {} %'.format(100 * correct / total))
# end evaluation
