import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

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

# Load model đã train
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Hàm dự đoán ảnh số viết tay
def predict_image(image_path):
    # Biến đổi ảnh về 28x28 grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # chuyển thành ảnh xám
        transforms.Resize((28, 28)),                   # resize 28x28
        transforms.ToTensor(),                         # đổi thành tensor [0,1]
        transforms.Normalize((0.5,), (0.5,))           # chuẩn hóa như MNIST
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print(f"Ảnh {image_path} được dự đoán là số: {predicted.item()}")

# Ví dụ chạy
predict_image("/mnt/c/Users/ngoc hieu/Downloads/anhthu.jpg")




