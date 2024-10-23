import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Creation Environement
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

# Creation de mon IA 
ia_model = 

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
    
    def forward(self, x):
        # Forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the network, loss function, and optimizer
input_size = 180
hidden_size = 5
output_size = 2
model = SimpleNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy input and target for demonstration
inputs = torch.randn(1, input_size)  # Batch size of 1, with 10 features
targets = torch.tensor([1])  # Target class (e.g., class 1)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)

# Backward pass and optimization
optimizer.zero_grad()  # Clear gradients
loss.backward()  # Backpropagation
optimizer.step()  # Update weights

print("Loss:", loss.item())



for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
