import sklearn
from sklearn import datasets
digits = datasets.load_digits()
type(digits)

#number of images
print(len(digits.images))

#number of labels
print(len(digits.target))

#type of images and target
print(type(digits.images))
print(type(digits.target))

#examine shape of images(metrix) and examine shape target
print(digits.images.shape)
print(digits.target.shape)

images = digits.images
labels = digits.target

img=images.reshape((images.shape[0], -1))
img.shape

digits.images[0]

import matplotlib.pyplot as plt
plt.gray() 
imgplot = plt.imshow(digits.images[0])
print("label: ",digits.target[0])
plt.show()

from sklearn import svm
classifier = svm.SVC(gamma=0.001,C=100.)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img, labels, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

classifier.fit(X_train, y_train)

score = classifier.score(X_test,y_test)
score

plt.gray() 
test_img = X_test[5].reshape(8,8)
imgplot = plt.imshow(test_img)
print("label: ",y_test[5])
plt.show()
t = X_test[5].reshape(1,-1)
pred = classifier.predict(t)
print("prediction: ",pred)


plt.gray() 
test_img = X_test[7].reshape(8,8)
imgplot = plt.imshow(test_img)
print("label: ",y_test[7])
plt.show()
t = X_test[7].reshape(1,-1)
pred = classifier.predict(t)
print("prediction: ",pred)


plt.gray() 
test_img = X_test[357].reshape(8,8)
imgplot = plt.imshow(test_img)
print("label: ",y_test[357])
plt.show()
t = X_test[357].reshape(1,-1)
pred = classifier.predict(t)
print("prediction: ",pred)

plt.gray() 
test_img = X_test[503].reshape(8,8)
imgplot = plt.imshow(test_img)
print("label: ",y_test[503])
plt.show()
t = X_test[503].reshape(1,-1)
pred = classifier.predict(t)
print("prediction: ",pred)