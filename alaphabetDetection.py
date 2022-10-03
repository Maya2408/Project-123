import numpy as np
import pandas as pd
import PIL.ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
#X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nClasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)
y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#draw a box?
roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

im_pil = Image.open(image)
image_bw = im_pil.convert('L')
image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
pixel_filter = 20

min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
max_pixel = np.max(image_bw_resized_inverted)
image_bw_resized_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
test_pred = clf.predict(test_sample)
print("Predicted class is: ", test_pred)
