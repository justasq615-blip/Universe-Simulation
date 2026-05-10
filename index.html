import numpy as np
import matplotlib.pyplot as plt

# --- parameetrid ---
alpha = 0.6   # tugevnemine
beta = 0.3    # hääbumine
dt = 0.01
T = 20
steps = int(T / dt)

# --- algvõrk (N sõlme) ---
N = 6
np.random.seed(1)
W = np.random.uniform(0.05, 0.2, (N, N))

# diagonaal nulliks (ei tugevda iseennast)
np.fill_diagonal(W, 0)

# salvestus
history = np.zeros((steps, N, N))

# --- simulatsioon ---
for t in range(steps):
    W2 = W @ W  # kaudsed teed (W^2)

    dW = alpha * W2 - beta * W

    W = W + dt * dW

    # vältida negatiivseid ja hoida stabiilsust
    W = np.clip(W, 0, 5)

    history[t] = W

# --- analüüs: keskmine tihedus ajas ---
density = history.mean(axis=(1,2))

plt.figure()
plt.plot(density)
plt.title("Võrgustiku keskmine tugevus ajas")
plt.xlabel("aeg")
plt.ylabel("keskmine W")
plt.grid()
plt.show()
