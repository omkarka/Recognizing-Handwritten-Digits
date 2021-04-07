# Recognizing-Handwritten-Digits

Recognizing handwritten text is a problem that can be traced back to the first automatic machines that needed to recognize individual characters in handwritten documents. 

An estimator that is useful in this case is sklearn.svm.SVC, which uses the technique of Support Vector Classification (SVC).
Thus, you have to import the svm module of the scikit-learn library. You can create an estimator of SVC type and then choose an initial setting, assigning the values C and
gamma generic values. These values can then be adjusted in a different way during the course of the analysis.

# from sklearn import svm
# svc = svm.SVC(gamma=0.001, C=100.)

The Digits Dataset:
The scikit-learn library provides numerous datasets that are useful for testing many problems of data analysis and prediction of the results. Also in this case there is a dataset
of images called Digits.
This dataset consists of 1,797 images that are 8x8 pixels in size.

Thus, you can load the Digits dataset into your Notebook.

# from sklearn import datasets
# digits = datasets.load_digits()

After loading the dataset, you can analyze the content. First, you can read lots of information about the datasets by calling the DESCR attribute.

# print(digits.DESCR)

Each dataset in the scikit-learn library has a field containing all the information.
The images of the handwritten digits are contained in a digits.images array. Each element of this array is an image that is represented by an 8x8 matrix of numerical values
that correspond to a grayscale from white, with a value of 0, to black, with the value 15.

# digits.images[0]

You can visually check the contents of this result using the matplotlib library.
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

The numerical values represented by images, i.e., the targets, are contained in the digit.targets array.
# digits.target

This dataset contains 1,797 elements, and so you can consider the first 1,791 as a
training set and will use the last six as a validation set.


