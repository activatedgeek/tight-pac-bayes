import numpy as np
from matplotlib import pyplot as plt

train_acc = 0.8621

levels = np.array([2, 3, 5, 7, 10, 15])
acc = np.array([0.6537, 0.7611, 0.8098, 0.8118, 0.8260, 0.8374])


plt.style.use('ggplot')
plt.figure(dpi=250)
plt.plot(levels, acc, label='Quantized Acc')
plt.scatter(levels, acc)
plt.axhline(y=train_acc, label=f'Train Acc {train_acc:2.2f}', color='blue')
plt.xlabel('# clusters')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for 5 epochs')
# plt.ylim([0.3, 1.0])
plt.show()

epochs = np.array([1, 3, 5, 7, 10])
acc = np.array([0.6997, 0.7160, 0.7654, 0.7448, 0.7941])

plt.figure(dpi=250)
plt.plot(epochs, acc, label='Quantized Acc')
plt.scatter(epochs, acc)
plt.axhline(y=train_acc, label=f'Train Acc {train_acc:2.2f}', color='blue')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for 3 clusters')
plt.show()
