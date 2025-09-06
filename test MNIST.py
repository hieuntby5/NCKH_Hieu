import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

# Định nghĩa lại model giống lúc train
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Load dữ liệu test
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Độ chính xác trên tập test: {:.2f}%".format(100 * correct / total))
