#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
st(student_grades, bins=range(0, 101, 10), edgecolor='black')  # Bins every 10 units, black outline
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.show()
