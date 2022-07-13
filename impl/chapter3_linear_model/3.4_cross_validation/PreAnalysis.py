import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="white", color_codes=True)
iris = sns.load_dataset("iris")

iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
sns.pairplot(iris, hue='species')
plt.show()
