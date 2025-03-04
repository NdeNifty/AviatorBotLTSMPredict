import requests
import matplotlib.pyplot as plt

response = requests.get('https://aviatorbotltsmpredict.onrender.com/performance')
data = response.json()

# Plot performance metrics
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['MAE', '% Within ±1.0'], [data['performance']['mae'], data['performance']['within_one_percent']])
plt.title('Performance Metrics')
plt.ylabel('Value')

# Plot learning curve
plt.subplot(1, 2, 2)
plt.plot(data['learning_curve'])
plt.title('Learning Curve (Loss History)')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()