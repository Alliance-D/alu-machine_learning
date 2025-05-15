#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
fruit_labels = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

x = np.arange(fruit.shape[1])
bottom = np.zeros(fruit.shape[1])
for i in range(fruit.shape[0]):
    plt.bar(x, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=fruit_labels[i])
    bottom += fruit[i]
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
