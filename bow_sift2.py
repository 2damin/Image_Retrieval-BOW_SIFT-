# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\pycharm_file\pyqt\bow_sift2.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1154, 840)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_1 = QtWidgets.QLabel(self.centralwidget)
        self.image_1.setGeometry(QtCore.QRect(30, 110, 311, 281))
        self.image_1.setAutoFillBackground(False)
        self.image_1.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_1.setObjectName("image_1")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(30, 70, 311, 41))
        self.label_1.setFrameShape(QtWidgets.QFrame.Box)
        self.label_1.setObjectName("label_1")
        self.image_2 = QtWidgets.QLabel(self.centralwidget)
        self.image_2.setGeometry(QtCore.QRect(400, 110, 311, 281))
        self.image_2.setAutoFillBackground(False)
        self.image_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_2.setObjectName("image_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 70, 311, 41))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setObjectName("label_2")
        self.image_3 = QtWidgets.QLabel(self.centralwidget)
        self.image_3.setGeometry(QtCore.QRect(770, 110, 311, 281))
        self.image_3.setAutoFillBackground(False)
        self.image_3.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_3.setObjectName("image_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(770, 70, 311, 41))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setObjectName("label_3")
        self.image_4 = QtWidgets.QLabel(self.centralwidget)
        self.image_4.setGeometry(QtCore.QRect(30, 470, 311, 281))
        self.image_4.setAutoFillBackground(False)
        self.image_4.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_4.setObjectName("image_4")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 430, 311, 41))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setObjectName("label_4")
        self.image_5 = QtWidgets.QLabel(self.centralwidget)
        self.image_5.setGeometry(QtCore.QRect(400, 470, 311, 281))
        self.image_5.setAutoFillBackground(False)
        self.image_5.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_5.setObjectName("image_5")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(400, 430, 311, 41))
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setObjectName("label_5")
        self.image_6 = QtWidgets.QLabel(self.centralwidget)
        self.image_6.setGeometry(QtCore.QRect(770, 470, 311, 281))
        self.image_6.setAutoFillBackground(False)
        self.image_6.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.image_6.setObjectName("image_6")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(770, 430, 311, 41))
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1154, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.image_1.setText(_translate("MainWindow", "ImageLabel"))
        self.label_1.setText(_translate("MainWindow", "TextLabel"))
        self.image_2.setText(_translate("MainWindow", "ImageLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.image_3.setText(_translate("MainWindow", "ImageLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.image_4.setText(_translate("MainWindow", "ImageLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.image_5.setText(_translate("MainWindow", "ImageLabel"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.image_6.setText(_translate("MainWindow", "ImageLabel"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))


    def show_image(self, img1, img2, img3, img4, img5, img6):
        self.label_1.setText("original_image")
        self.label_2.setText("1st Predict")
        self.label_3.setText("2nd Predict")
        self.label_4.setText("3rd Predict")
        self.label_5.setText("4th Predict")
        self.label_6.setText("5th Predict")
        image1 = self.img_resize(img1)
        image2 = self.img_resize(img2)
        image3 = self.img_resize(img3)
        image4 = self.img_resize(img4)
        image5 = self.img_resize(img5)
        image6 = self.img_resize(img6)
        #pixmap = QPixmap(image)
        #pixmap2 = pixmap.scaled(311,280, QtCore.Qt.KeepAspectRatio)
        self.image_1.setPixmap(image1)
        self.image_2.setPixmap(image2)
        self.image_3.setPixmap(image3)
        self.image_4.setPixmap(image4)
        self.image_5.setPixmap(image5)
        self.image_6.setPixmap(image6)
        #self.image_1.resize(pixmap.width(), pixmap.height())
        #self.show()

    def img_resize(self, image):
        pixmap = QPixmap(image)
        pixmap2 = pixmap.scaled(311, 280, QtCore.Qt.KeepAspectRatio)

        return pixmap2


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

