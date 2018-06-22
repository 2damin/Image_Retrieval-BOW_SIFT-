import cv2
import numpy as np
import time
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans, KMeans
# Local dependencies

import constants
import descriptors
import filenames
import utils


class Classifier:
    """
    Class for making training and testing in image classification.
    """
    def __init__(self, dataset, log):
        """
        Initialize the classifier object.
        Args:
            dataset (Dataset): The object that stores the information about the dataset.
            log (Log): The object that stores the information about the times and the results of the process.

        Returns:
            void
        """
        self.dataset = dataset
        self.log = log

    def train(self, svm_kernel, k, des_name, test_image, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the training set and then calculates the SVM for them.

        Args:
            svm_kernel (constant): The kernel of the SVM that will be created.
            codebook (NumPy float matrix): Each row is a center of a codebook of Bag of Words approach.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            cv2.SVM: The Support Vector Machine obtained in the training phase.
        """
        isTrain= True
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        x_filename = filenames.vlads_train(k, des_name)
        print("Getting global descriptors for the training set.")
        start = time.time()
        x, y, x_test, y1, testimg_h, cluster_model = self.get_data_and_labels(self.dataset.get_train_set(), test_image,
                                                                   KMeans(n_clusters=k), k, des_name ,isTrain,des_option,isTrain)
        utils.save(x_filename, x)
        end = time.time()
        svm_filename = filenames.svm(k, des_name, svm_kernel)
        utils.save(svm_filename, x)
        print("Calculating the Support Vector Machine for the training set...")

        return x, y, x_test, y1, testimg_h, cluster_model
        #svm, cluster_model

    def test(self, x, y, x_test, y1, testimg_h, cluster_model, k, des_option = constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the testing set and use the svm given as a parameter to predict all the elements

        Args:
            codebook (NumPy matrix): Each row is a center of a codebook of Bag of Words approach.
            svm (cv2.SVM): The Support Vector Machine obtained in the training phase.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            NumPy float array: The result of the predictions made.
            NumPy float array: The real labels for the testing set.
        """
        svmL = svm.SVC(C=1.0, kernel="linear")
        isTrain = False
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        print("Getting global descriptors for the testing set...")
        #start = time.time()
        #x, y, cluster_model= self.get_data_and_labels_test(self.dataset.get_test_set(), cluster_model, k, des_name,isTrain,des_option)
        #end = time.time()
        start = time.time()
        svmL.fit(x, y)
        result = svmL.predict(x_test)
        end = time.time()
        self.log.predict_time(end - start)
        mask = result == y1
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        self.log.accuracy(accuracy)
        svm_result = svmL.predict(testimg_h)
        #svm_score = svmL.score(x_test, y1)

        #score = svmL.score(testimg_h,y)
        #print("score of test : \n", svm_score)
        return result, y1, svm_result

    def get_data_and_labels(self, img_set, test_image, cluster_model, k, des_name, codebook,isTrain, des_option = constants.ORB_FEAT_OPTION):
        """
        Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
        descriptor for each image and stores the label with the class of the image.
        Args:
            img_set (string array): The list of image paths for the set.
            codebook (numpy float matrix): Each row is a center and each column is a dimension of the centers.
            des_option (integer): The option of the feature that is going to be used as local descriptor.

        Returns:
            NumPy float matrix: Each row is the global descriptor of an image and each column is a dimension.
            NumPy float array: Each element is the number of the class for the corresponding image.
        """
        y = []
        x = None
        img_descs = []
        
        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]
            
            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                img = cv2.imread(img_paths[i])
                
                des,y = descriptors.sift(img,img_descs,y,class_number)
                """ des : descriptor
                    y : class_number
                """
        isTrain = int(isTrain)


        X, X_test, y1, testimg_hist, cluster_model = descriptors.cluster_features(des, test_image, cluster_model=KMeans(n_clusters=128))
        """
        X: histogram of words
        cluster_model : MiniBatchKMeans
        testimg_hist : testImage histogram
        """
        """
        if isTrain == 1:
        else:
            X = descriptors.img_to_vect(des,cluster_model)
        """

        print('X',X.shape,X)
        y = np.float32(y)[:,np.newaxis]
        x = np.matrix(X)
        y1 = np.float32(y1)[:, np.newaxis]
        x_test = np.matrix(X_test)
        testimg_h = np.matrix(testimg_hist)
        return x, y, x_test, y1, testimg_h, cluster_model

    def get_data_and_labels_test(self, img_set, cluster_model, k, des_name, codebook, isTrain,
                            des_option=constants.ORB_FEAT_OPTION):
        """
        Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
        descriptor for each image and stores the label with the class of the image.
        Args:
            img_set (string array): The list of image paths for the set.
            codebook (numpy float matrix): Each row is a center and each column is a dimension of the centers.
            des_option (integer): The option of the feature that is going to be used as local descriptor.

        Returns:
            NumPy float matrix: Each row is the global descriptor of an image and each column is a dimension.
            NumPy float array: Each element is the number of the class for the corresponding image.
        """
        y = []
        x = None
        img_descs = []

        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]

            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                img = cv2.imread(img_paths[i])

                des, y = descriptors.sift(img, img_descs, y, class_number)
                """ des : descriptor
                    y : class_number
                """
        isTrain = int(isTrain)

        X, cluster_model = descriptors.cluster_test(des, cluster_model=MiniBatchKMeans(n_clusters=128))
        """
        X: histogram of words
        cluster_model : MiniBatchKMeans
        """
        """
        if isTrain == 1:
        else:
            X = descriptors.img_to_vect(des,cluster_model)
        """

        print('X', X.shape, X)
        y = np.float32(y)[:, np.newaxis]
        x = np.matrix(X)
        return x, y, cluster_model