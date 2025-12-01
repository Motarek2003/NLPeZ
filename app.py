import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("Python version:", __import__("platform").python_version())
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ---- NumPy demo ----
print("\nNumPy demo:")
arr = np.array([1, 2, 3, 4])
print("Array:", arr)
print("Mean:", arr.mean())

# ---- Pandas demo ----
print("\nPandas demo:")
df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [2, 4, 6, 8]
})
print(df)

# ---- scikit-learn demo ----
print("\nscikit-learn demo:")
X = df[["x"]]
y = df["y"]

model = LinearRegression().fit(X, y)
print("Predicted y for x=5:", model.predict([[5]])[0])

# ---- PyTorch demo ----
print("\nPyTorch demo:")
tensor = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", tensor)
print("Tensor * 3:", tensor * 3)

# Simple linear model in PyTorch
linear = torch.nn.Linear(1, 1)
x_tensor = torch.tensor([[1.0], [2.0], [3.0]])
y_tensor = torch.tensor([[2.0], [4.0], [6.0]])

optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for step in range(50):
    pred = linear(x_tensor)
    loss = loss_fn(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Torch model output for x=5:", linear(torch.tensor([[5.0]])).item())
