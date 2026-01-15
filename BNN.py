import numpy as np
import torch
from torchvision import datasets, transforms

# 1. LOAD AND PREPROCESS MNIST
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_images = train_dataset.data.numpy().astype(np.float32)
test_images  = test_dataset.data.numpy().astype(np.float32)
train_labels = np.array(train_dataset.targets)
test_labels  = np.array(test_dataset.targets)

# Normalize về [-1, 1]
train_images = (train_images / 255 - 0.5) / 0.5 # shape (60000,28,28)
test_images  = (test_images / 255 - 0.5) / 0.5 # shape (10000,28,28)

# Flatten
x_train = train_images.reshape(len(train_images), -1) # reshape(60000,784)
x_test  = test_images.reshape(len(test_images), -1)

# 2. DEFINE BNN
class SignActivation:
    def forward(self, x):
        self.x = x
        return np.where(self.x >= 0, 1, -1)
    def backward(self, grad_output, mode, layer):
        # Straight-Through Estimator
        grad_input = grad_output * (np.abs(self.x) <= 25)  # Ở ĐÂY, TA CHỌN NGƯỠNG (THRESHOLD LÀ 25, ĐƯỢC XEM NHƯ LÀ "VÙNG CHO PHÉP GRADIENT ĐI QUA")
        saturated = np.sum(np.abs(self.x) > 25)
        total = self.x.size
        if mode == 1:
          print(f"Saturated ratio in sign layer {layer}: {saturated/total:.2%}")
         # print(f"Input: {np.round(self.x.flatten())}")
          print("\n")
        return grad_input

class FcLayer:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features) * 0.1
        self.bias = np.zeros((1, out_features))
    def forward(self, x):
        self.x = x
        self.binary_weight = np.where(self.weight >= 0, 1, -1)
        return np.dot(x ,self.binary_weight.T) + self.bias
    def backward(self, grad_output, lr):
        grad_w = np.dot(grad_output.T, self.x)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.binary_weight)
        # Update real weights
        self.weight -= lr * grad_w
        self.bias -= lr * grad_b
        self.weight = np.clip(self.weight, -1, 1)
        return grad_input

class BNN:
    def __init__(self):
        self.sign0 = SignActivation()
        self.fc1 = FcLayer(784, 512)
        self.sign1 = SignActivation()
        self.fc2 = FcLayer(512, 10)
        self.sign2 = SignActivation()
    def forward(self, x):
        x = self.sign0.forward(x)
        x = self.fc1.forward(x)
        x = self.sign1.forward(x)
        x = self.fc2.forward(x)
        x = self.sign2.forward(x)
        return x
    def backward(self, grad_output, lr, epoch, mode):
        grad = self.sign2.backward(grad_output, mode, 2)
        grad = self.fc2.backward(grad, lr)
        grad = self.sign1.backward(grad, mode, 1)
        grad = self.fc1.backward(grad, lr)
        grad = self.sign0.backward(grad, 0, 0)
        return grad

# 3. LOSS FUNCTION
def mse_loss(pred, target):
    loss = np.mean((pred - target) ** 2)
    grad = 2 * (pred - target) / target.shape[0]
    return loss, grad

def label_to_binary(y, num_classes=10):
    batch_size = y.shape[0]
    binary = -np.ones((batch_size, num_classes))
    binary[np.arange(batch_size), y] = 1
    return binary

# ======================================================
# 4. TRAINING
# ======================================================
model = BNN()
lr = 0.001
epochs = 25
batch_size = 64
best_acc = 0

for epoch in range(epochs):
    total_loss = 0
    correct = 0

    # ĐẢO CÁC BỨC ẢNH
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    train_labels = train_labels[idx]

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = train_labels[i:i+batch_size]
        y_bin = label_to_binary(y_batch)

        # Forward
        out = model.forward(x_batch)
        loss, grad = mse_loss(out, y_bin)
        total_loss += loss

        print_flag = (i == 0) or (i + batch_size >= len(x_train))  # đầu và cuối thôi

        # Backward
        model.backward(grad, lr, epoch, print_flag)

        # Accuracy
        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == y_batch)

    acc = correct / len(x_train) * 100
    print(f"Epoch {epoch+1:02d}| Loss: {loss:.2f} Train Acc: {acc:.2f}%")

  # ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU DÙNG ĐỂ TEST
    correct_test = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = test_labels[i:i+batch_size]
        out = model.forward(x_batch)
        preds = np.argmax(out, axis=1)
        correct_test += np.sum(preds == y_batch)
    acc_test = correct_test / len(x_test) * 100
     # SO SÁNH GIÁ TRỊ
    if acc_test > best_acc:
        best_acc = acc_test

    print(f"Test Accuracy: {acc_test:.2f}%\n")

print(f"Best Test Accuracy: {best_acc:.2f}%")
