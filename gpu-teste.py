import torch
import torch.nn as nn
import torch.optim as optim

# 1. Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")

if device.type == "cuda":
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória Total da GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 2. Criar um modelo simples
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Rede linear bem simples

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().to(device)

# 3. Gerar dados de treino aleatórios
X = torch.randn(1000, 10).to(device)
y = torch.randn(1000, 1).to(device)

# 4. Definir otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 5. Treinar rapidamente
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Checar novamente o uso de memória
if device.type == "cuda":
    print(f"Memória utilizada após treino: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")

print("Teste de GPU concluído!")
