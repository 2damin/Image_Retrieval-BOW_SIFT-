import cv2
import numpy as np
import time
import os
import sys
import copy
sys.path.insert(0, "D:\\pycharm_file\\bow_sift")

# Local dependencies
from classifier import Classifier
from dataset import Dataset
import descriptors
import constants
import utils
import filenames
from bow_sift import bow_sift2
from log import Log
import cv2
from matplotlib.image import imread
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_log_error

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap



def main(is_interactive=True, k=128, des_option=constants.ORB_FEAT_OPTION, svm_kernel=cv2.ml.SVM_LINEAR):
    if not is_interactive:
        experiment_start = time.time()
    # Check for the dataset of images
    if not os.path.exists(constants.DATASET_PATH):
        print("Dataset not found, please copy one.")
        return
    dataset = Dataset(constants.DATASET_PATH)
    dataset.generate_sets()

    # Check for the directory where stores generated files
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)

    if is_interactive:
        des_option = input("[1] for using ORB features or [2] to use SIFT features.\n")
        k = input("the number of cluster centers you want for the codebook.\n")
        svm_option = input("[1]SVM kernel Linear or [2]RBF.\n")
        test_image_dir = input("Input test image.\n")
        svm_kernel = cv2.ml.SVM_LINEAR if svm_option == 1 else cv2.ml.SVM_RBF

    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    print(des_name)
    log = Log(k, des_name, svm_kernel)
    test_image = cv2.imread(test_image_dir)

    codebook_filename = filenames.codebook(k, des_name)
    print('codebook_filename')
    print(codebook_filename)
    start = time.time()   
    end = time.time()
    log.train_des_time(end - start)
    start = time.time()
    end = time.time()
    log.codebook_time(end - start)
    # Train and test the dataset
    classifier = Classifier(dataset, log)
    x, y, x_test, y1, testimg_h, cluster_model = classifier.train(svm_kernel, k, des_name, test_image, des_option=des_option, is_interactive=is_interactive)
    print("Training ready. Now beginning with testing")
    #utils.show_conf_mat(x_test)
    print("x : \n", x)
    print("\n", np.shape(x))
    #show_histogram(x_test, 128, 0)
    #show_histogram(x_test,1)

    result, labels, svm_result = classifier.test(x, y, x_test, y1, testimg_h, cluster_model, k, des_option=des_option, is_interactive=is_interactive)
    print('test result')
    print(result)
    label = np.ndarray.flatten(labels)
    print(label)
    print("predict test image:\n")
    print(svm_result)

    test_accuracy = accuracy_score(result, label)
    print("Test accuracy : ", test_accuracy)
    # Store the results from the test
    classes = dataset.get_classes()
    log.classes(classes)
    log.classes_counts(dataset.get_classes_counts())
    result_filename = filenames.result(k, des_name, svm_kernel)
    test_count = len(dataset.get_test_set()[0])
    result_matrix = np.reshape(result, (len(classes), test_count))
    utils.save_csv(result_filename, result_matrix)

    # Create a confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for i in range(len(result)):
        predicted_id = int(result[i])
        real_id = int(labels[i])
        confusion_matrix[real_id][predicted_id] += 1

    print("Confusion Matrix =\n{0}".format(confusion_matrix))
    log.confusion_matrix(confusion_matrix)
    log.save()
    print("Log saved on {0}.".format(filenames.log(k, des_name, svm_kernel)))
    if not is_interactive:
        experiment_end = time.time()
        elapsed_time = utils.humanize_time(experiment_end - experiment_start)
        print("Total time during the experiment was {0}".format(elapsed_time))
    else:
        # Show a plot of the confusion matrix on interactive mode
        utils.show_conf_mat(confusion_matrix)
        #raw_input("Press [Enter] to exit ...")

    ranking_img = []
    print(np.shape(testimg_h[0,:]))
    print(np.shape(x[0,:]))
    print(svm_result[0])
    for i in range(90):
        j = i + int(svm_result[0])*90
        ranking_img.append(mean_squared_log_error(testimg_h[0,:], x[j,:]))

    print("ranking : \n")
    print(ranking_img)
    ranking_img_origin = copy.deepcopy(ranking_img)
    ranking_img.sort()
    print(ranking_img_origin)
    index1 = ranking_img.index(ranking_img_origin[0])
    index2 = ranking_img.index(ranking_img_origin[1])
    index3 = ranking_img.index(ranking_img_origin[2])
    index4 = ranking_img.index(ranking_img_origin[3])
    index5 = ranking_img.index(ranking_img_origin[4])

    if svm_result[0] == 0:
        folder = "003_00"
    elif svm_result[0] == 1:
        folder = "002_00"
    elif svm_result[0] == 2:
        folder = "129_00"
    elif svm_result[0] == 3:
        folder ="098_00"
    elif svm_result[0] == 4:
        folder ="145_00"
    elif svm_result[0] == 5:
        folder ="158_00"
    elif svm_result[0] == 6:
        folder ="178_00"
    elif svm_result[0] == 7:
        folder ="211_00"
    elif svm_result[0] == 8:
        folder ="213_00"
    elif svm_result[0] == 9:
        folder ="250_00"
    elif svm_result[0] == 10:
        folder ="252_00"


    image1_name = "D:\\pycharm_file\\bow_sift\\dataset_real2\\" + str(int(svm_result[0] + 1)) + "\\" + folder + str(
        int(index1+10)) +".jpg"
    image2_name = "D:\\pycharm_file\\bow_sift\\dataset_real2\\" + str(int(svm_result[0] + 1)) + "\\" + folder + str(
        int(index2 + 10)) + ".jpg"
    image3_name = "D:\\pycharm_file\\bow_sift\\dataset_real2\\" + str(int(svm_result[0] + 1)) + "\\" + folder + str(
        int(index3 + 10)) + ".jpg"
    image4_name = "D:\\pycharm_file\\bow_sift\\dataset_real2\\" + str(int(svm_result[0] + 1)) + "\\" + folder + str(
        int(index4 + 10)) + ".jpg"
    image5_name = "D:\\pycharm_file\\bow_sift\\dataset_real2\\" + str(int(svm_result[0] + 1)) + "\\" + folder + str(
        int(index5 + 10)) + ".jpg"
    print(image1_name)
    print(index1)
    #show_histogram(testimg_h, int(k), 0)
    #show_histogram(x, int(k), int(svm_result[0]*140) + index1)

    show_result(test_image_dir, image1_name, image2_name, image3_name, image4_name, image5_name)

"""
def show_histogram(x_test, n_cluster, image):
    x_histogram = []

    for i in range(n_cluster):
        x_histogram.append(x_test[image, i])
    index = np.arange(n_cluster)
    #print(np.shape(x_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set(xlim=[0.,n_cluster.] , ylim=[0.,60.])
    plt.bar(index,x_histogram)
    plt.title("Test data histogram")
    plt.xlabel("center of clusters")
    plt.ylabel("Frequency")
    plt.show()
"""
def show_result(origin_dir, img1, img2, img3, img4, img5):
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    origin_img = QPixmap(origin_dir)
    image1 = QPixmap(img1)
    image2 = QPixmap(img2)
    image3 = QPixmap(img3)
    image4 = QPixmap(img4)
    image5 = QPixmap(img5)
    #img_np = np.reshape(img)
    gui = bow_sift2.Ui_MainWindow()
    gui.setupUi(MainWindow)
    gui.show_image(origin_img, image1, image2, image3, image4, image5)
    MainWindow.show()
    sys.exit(app.exec_())




if __name__ == '__main__':
    main()