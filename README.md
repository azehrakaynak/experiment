# experiment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
x = np.array([0.089, 0.165, 0.232, 0.294])
y = np.array([50, 100, 150, 200])
plt.plot(x, y)
plt.xlabel("F(N)")
plt.ylabel("∆x(meter)")
plt.show()

linreg = LinearRegression()
x = x.reshape(-1, 1)
linreg.fit(x, y)
y_pred = linreg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color="red")
plt.show()
x = [0.089, 0.165, 0.232, 0.294]
y = [50, 100, 150, 200]

slope, intercept, r_value, p_value, std_err = linregress(x, y)
print(slope)
