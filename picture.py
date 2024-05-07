import numpy as np
import matplotlib.pyplot as plt

def custom_activation(x):
    return np.maximum(0, x) * (1 / (1 + np.exp(-x)))

# Generate some data
x = np.linspace(-10, 10, 400)
y = custom_activation(x)

# Plot the function
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='ReLU(x) * Sigmoid(x)')
plt.title('Custom Activation Function')
plt.xlabel('x')
plt.ylabel('Activation Output')
plt.legend()
plt.grid(True)
plt.show()
