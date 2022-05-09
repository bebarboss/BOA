from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint, QRect,
                          QSize, QUrl, Qt, QPropertyAnimation, QThread, pyqtSignal, pyqtSlot, QTimer)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase,
                         QIcon, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient, QImage)
from PyQt5.QtWidgets import *
from splash_screen import Ui_SplashScreen
from main import Ui_MainWindow


import sys
import cv2
import numpy as np
import os
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import sklearn
import time
from datetime import datetime
from PIL.ImageQt import ImageQt
from PIL import Image


# Globals
counter = 0
GLOBAL_STATE = 0


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('main.ui', self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.toggle_button.clicked.connect(self.ToggleMenu)
        self.ui.pushButton.clicked.connect(self.close)

        

        def moveWindow(event):
            # Move window if left mouse button is clicked
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()
        self.ui.title_frame.mouseMoveEvent = moveWindow

    # PAGES
        ###########################################################################################################################################
        # PAGE 1
        self.ui.recognition_page.clicked.connect(
            lambda: self.ui.pages.setCurrentWidget(self.ui.page_2))
        self.ui.recognition_page.clicked.connect(
            lambda: self.ui.label_7.setText("recognition"))
        self.ui.open_cam.clicked.connect(self.open_img2)
        self.ui.pushButton_13.clicked.connect(self.take_attendance_image)

        # PAGE 2
        self.ui.registration_page.clicked.connect(
            lambda: self.ui.pages.setCurrentWidget(self.ui.page_3))
        self.ui.registration_page.clicked.connect(
            lambda: self.ui.label_7.setText("registration"))
        self.ui.pushButton_3.clicked.connect(self.open_img)
        self.ui.pushButton_8.clicked.connect(self.generate_train_dataset)
        self.ui.pushButton_12.clicked.connect(self.generate_test_dataset)
        self.ui.pushButton_9.clicked.connect(self.train_model)

        # PAGE 3
        self.ui.attendance_page.clicked.connect(
            lambda: self.ui.pages.setCurrentWidget(self.ui.page_4))
        self.ui.attendance_page.clicked.connect(
            lambda: self.ui.label_7.setText("info"))
        ###########################################################################################################################################

    def open_img(self):
        self.fname = QtWidgets.QFileDialog.getOpenFileName(
            filter="Image (*.*)")[0]
        image = cv2.imread(self.fname, 0)
        path = 'E:/BOA/data/Images for Attendance'
        images = [1]
        image_numbers = [1]
        myList = os.listdir(path)
        for img in myList:
            curImg = cv2.imread(f'{path}/{img}')
            images.append(curImg)
            image_numbers.append(os.path.splitext(img)[0])
        img_id = int(image_numbers[-1])
        self.tmp = image
        imagenames = path+"/"+str(img_id)+".jpg"
        cv2.imwrite(imagenames, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QImage(image, image.shape[1],
                     image.shape[0], QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QPixmap.fromImage(img))
        self.ui.label_2.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def setPhoto(self):
        image = cv2.imread('E:/BOA/data/Images for Attendance/1.jpg', 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QImage(image, image.shape[1],
                     image.shape[0], QImage.Format_RGB888)
        self.ui.label_2.setPixmap(QPixmap.fromImage(img))
        self.ui.label_2.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    ############################################################################################################
        ###########################################################################################################################################

    def generate_train_dataset(self):
        host = self.ui.lineEdit.text()
        path1 = 'E:/BOA/data/train'
        path2 = os.path.join(path1, host)
        if not os.path.exists(path2):
            os.mkdir(path2)
        else:
            pass
        img_counter = 0
        r = 0
        image = cv2.imread('E:/BOA/data/Images for Attendance/1.jpg', 0)
        while True:
            adjusted = cv2.convertScaleAbs(image, alpha=1.7, beta=9)
            fil = cv2.bilateralFilter(adjusted, 10, 25, 50)
            blur = cv2.medianBlur(fil, 15)
            img_Th = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
            blur2 = cv2.medianBlur(img_Th, 13)
            kernel_img = np.ones((5, 5), np.uint8)
            open_img = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, kernel_img)
            New_image = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(New_image, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(
                gray, 225, 255, cv2.THRESH_BINARY_INV)
            contour, _ = cv2.findContours(
                image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i in contour:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    mask = np.zeros(New_image.shape, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), 170, (255, 255, 255), -1)
                    ROI = cv2.bitwise_and(New_image, mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = cv2.boundingRect(mask)
                    result = ROI[y:y+h, x:x+w]
                    mask = mask[y:y+h, x:x+w]
                    result[mask == 22] = (0, 0, 0)

            if r < 360:
                r = r+3.6
                (h, w) = result.shape[:2]
                center = (w / 2, h / 2)
                scale = 1

                print('collecting {}'.format(r))
                for imgnum in range(1):

                    m = cv2.getRotationMatrix2D(center, r, scale)
                    rotated = cv2.warpAffine(result, m, (w, h))
                    imagename = path2+"/"+host+"."+str(r)+".jpg"
                    cv2.imwrite(imagename, rotated)
                    img_counter += 1
                    if img_counter == 100:
                        self.setPhoto()
                        break
            cv2.waitKey()
        

    def generate_test_dataset(self):
        host = self.ui.lineEdit.text()
        path1 = 'E:/BOA/data/test'
        path2 = os.path.join(path1, host)
        if not os.path.exists(path2):
            os.mkdir(path2)
        else:
            pass
        img_counter = 0
        r = 0
        image = cv2.imread('E:/BOA/data/Images for Attendance/1.jpg', 0)
        while True:
            adjusted = cv2.convertScaleAbs(image, alpha=1.7, beta=9)
            fil = cv2.bilateralFilter(adjusted, 10, 25, 50)
            blur = cv2.medianBlur(fil, 15)
            img_Th = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
            blur2 = cv2.medianBlur(img_Th, 13)
            kernel_img = np.ones((5, 5), np.uint8)
            open_img = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, kernel_img)
            New_image = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(New_image, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(
                gray, 225, 255, cv2.THRESH_BINARY_INV)
            contour, _ = cv2.findContours(
                image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i in contour:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    mask = np.zeros(New_image.shape, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), 170, (255, 255, 255), -1)
                    ROI = cv2.bitwise_and(New_image, mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = cv2.boundingRect(mask)
                    result = ROI[y:y+h, x:x+w]
                    mask = mask[y:y+h, x:x+w]
                    result[mask == 22] = (0, 0, 0)

            if r < 360:
                r = r+3.6
                (h, w) = result.shape[:2]
                center = (w / 2, h / 2)
                scale = 1

                print('collecting {}'.format(r))
                for imgnum in range(1):

                    m = cv2.getRotationMatrix2D(center, r, scale)
                    rotated = cv2.warpAffine(result, m, (w, h))
                    imagename = path2+"/"+host+"."+str(r)+".jpg"
                    cv2.imwrite(imagename, rotated)
                    img_counter += 1
                if img_counter == 100:
                    self.setPhoto()
                    break

    def train_model(self):
        # Specifying the folder where images are present
        TrainingImagePath = 'E:/BOA/data/train'

        # Defining pre-processing transformations on raw images of training data
        # These hyper parameters helps to generate slightly twisted versions
        # of the original image, which leads to a better model, since it learns
        # on the good and bad mix of images

        train_datagen = ImageDataGenerator(rescale=1./255,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True)

        # Defining pre-processing transformations on raw images of testing data
        # No transformations are done on the testing images
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Generating the Training Data
        training_set = train_datagen.flow_from_directory(
            TrainingImagePath,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

        # Generating the Testing Data
        test_set = test_datagen.flow_from_directory(
            TrainingImagePath,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')

        # Printing class labels for each face
        test_set.class_indices

        # class_indices have the numeric tag for each face
        TrainClasses = training_set.class_indices

        # Storing the face and the numeric tag for future reference
        ResultMap = {}
        for faceValue, handName in zip(TrainClasses.values(), TrainClasses.keys()):
            ResultMap[faceValue] = handName

        # Saving the face map for future reference
        with open("ResultsMap.pkl", 'wb') as fileWriteStream:
            pickle.dump(ResultMap, fileWriteStream)

        # The model will give answer as a numeric tag
        # This mapping will help to get the corresponding face name for it
        print("Mapping of hand and its ID", ResultMap)

        # The number of neurons for the output layer is equal to the number of faces
        OutputNeurons = len(ResultMap)
        print('\n The Number of output neurons: ', OutputNeurons)

        '''######################## Create CNN deep learning model ########################'''
        '''Initializing the Convolutional Neural Network'''
        classifier = Sequential()

        ''' STEP--1 Convolution
        # Adding the first layer of CNN
        # we are using the format (64,64,3) because we are using TensorFlow backend
        # It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
        '''
        classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(
            1, 1), input_shape=(64, 64, 3), activation='relu'))

        '''# STEP--2 MAX Pooling'''
        classifier.add(MaxPool2D(pool_size=(2, 2)))

        '''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
        classifier.add(Convolution2D(64, kernel_size=(
            5, 5), strides=(1, 1), activation='relu'))

        classifier.add(MaxPool2D(pool_size=(2, 2)))

        '''# STEP--3 FLattening'''
        classifier.add(Flatten())

        '''# STEP--4 Fully Connected Neural Network'''
        classifier.add(Dense(64, activation='relu'))

        classifier.add(Dense(OutputNeurons, activation='softmax'))

        '''# Compiling the CNN'''
        #classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=["accuracy"])

        classifier.summary()

        # Measuring the time taken by the model to train
        StartTime = time.time()

        # Starting the model training
        classifier.fit_generator(
            training_set,
            steps_per_epoch=len(training_set),
            epochs=30,
            validation_data=test_set,
            validation_steps=len(test_set))

        EndTime = time.time()
        print("###### Total Time Taken: ", round(
            (EndTime-StartTime)/60), 'Minutes ######')

        # Save the model
        classifier.save('E:/BOA/data/model_new')

    def open_img2(self):
        self.fname = QtWidgets.QFileDialog.getOpenFileName(
            filter="Image (*.*)")[0]
        image = cv2.imread(self.fname, 0)
        path = 'E:/BOA/data/Images for Attendance'
        path2 = 'E:/BOA/data/Images for Attendance2'
        images = [1]
        image_numbers = [1]
        myList = os.listdir(path)
        for img in myList:
            curImg = cv2.imread(f'{path}/{img}')
            images.append(curImg)
            image_numbers.append(os.path.splitext(img)[0])
        img_id = int(image_numbers[-1])
        self.tmp = image
        imagenames = path+"/"+str(img_id)+".jpg"
        cv2.imwrite(imagenames, image)
        imagenames = path2+"/"+str(img_id)+".jpg"
        cv2.imwrite(imagenames, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QImage(image, image.shape[1],
                     image.shape[0], QImage.Format_RGB888)
        self.ui.label_13.setPixmap(QPixmap.fromImage(img))
        self.ui.label_13.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def setPhoto2(self):
        image = cv2.imread('E:/BOA/data/Images for Attendance2/1.jpg', 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QImage(image, image.shape[1],
                     image.shape[0], QImage.Format_RGB888)
        self.ui.label_13.setPixmap(QPixmap.fromImage(img))
        self.ui.label_13.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def take_attendance_image(self):
        path = 'E:/BOA/data/Images for Attendance'
        images = [1]
        image_numbers = [1]
        myList = os.listdir(path)
        for img in myList:
            curImg = cv2.imread(f'{path}/{img}')
            images.append(curImg)
            image_numbers.append(os.path.splitext(img)[0])
        img_id = int(image_numbers[-1])
        images = cv2.imread('E:/BOA/data/Images for Attendance/1.jpg', 0)
        while True:
            adjusted = cv2.convertScaleAbs(images, alpha=1.7, beta=9)
            fil = cv2.bilateralFilter(adjusted, 10, 25, 50)
            blur = cv2.medianBlur(fil, 15)
            img_Th = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
            blur2 = cv2.medianBlur(img_Th, 13)
            kernel_img = np.ones((5, 5), np.uint8)
            open_img = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, kernel_img)
            New_image = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(New_image, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(
                gray, 225, 255, cv2.THRESH_BINARY_INV)
            contour, _ = cv2.findContours(
                images, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i in contour:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    mask = np.zeros(New_image.shape, dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), 170, (255, 255, 255), -1)
                    ROI = cv2.bitwise_and(New_image, mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = cv2.boundingRect(mask)
                    result = ROI[y:y+h, x:x+w]
                    mask = mask[y:y+h, x:x+w]
                    result[mask == 22] = (0, 0, 0)
                    imagenames = path+"/"+str(img_id)+".jpg"
                    cv2.imwrite(imagenames, result)
            model = keras.models.load_model('E:/BOA/data/model_new')

            # Specifying the folder where images are present
            TrainingImagePath = 'E:/BOA/data/train'
            # Defining pre-processing transformations on raw images of training data
            # These hyper parameters helps to generate slightly twisted versions
            # of the original image, which leads to a better model, since it learns
            # on the good and bad mix of images
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               shear_range=0.1,
                                               zoom_range=0.1,
                                               horizontal_flip=True)
            # Defining pre-processing transformations on raw images of testing data
            # No transformations are done on the testing images
            test_datagen = ImageDataGenerator(rescale=1./255,)
            # Generating the Training Data
            training_set = train_datagen.flow_from_directory(
                TrainingImagePath,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
            # Generating the Testing Data
            test_set = test_datagen.flow_from_directory(
                TrainingImagePath,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
            # Printing class labels for each face
            test_set.class_indices
            # class_indices have the numeric tag for each face
            TrainClasses = training_set.class_indices
            # Storing the face and the numeric tag for future reference
            ResultMap = {}
            for handValue, handName in zip(TrainClasses.values(), TrainClasses.keys()):
                ResultMap[handValue] = handName
                # Prediction
            my_img_path = 'E:/BOA/data/Images for Attendance'
            myList2 = os.listdir(my_img_path)
            for i in myList2:
                if img_id == int(os.path.splitext(i)[0]):
                    image_path = my_img_path+"/"+str(img_id)+".jpg"
                    break
            test_image = image.load_img(image_path, target_size=(64, 64))
            test_image = image.img_to_array(test_image)

            test_image = np.expand_dims(test_image, axis=0)
            

            result = model.predict(test_image, verbose=0)
            print(result)
            print(ResultMap[np.argmax(result)])
            
            self.ui.label_person_name.setText(ResultMap[np.argmax(result)])

            if cv2.waitKey(1):
                self.setPhoto2()
                break

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
# toggle
        ###########################################################################################################################################

    def ToggleMenu(self):
        if True:

            # Get width
            width = self.ui.frame_left_menu.width()
            maxExtend = 200
            standard = 70

            # Set max width
            if width == 70:
                widthExtended = maxExtend
                self.ui.label_8.setText("Regconition")
                self.ui.label_9.setText("registration")
                self.ui.label_10.setText("attendance")
            else:
                widthExtended = standard
                self.ui.label_8.setText("")
                self.ui.label_9.setText("")
                self.ui.label_10.setText("")

            # Animation
            self.animation = QPropertyAnimation(
                self.ui.frame_left_menu, b"minimumWidth")
            self.animation.setDuration(400)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
            self.animation.start()
        ###########################################################################################################################################


class SplashScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(SplashScreen, self).__init__()
        uic.loadUi('splash_screen.ui', self)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)

        # Remove title bar
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Drop shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 200))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        self.timer.start(40)

        # Change loading text
        QtCore.QTimer.singleShot(
            500, lambda: self.ui.label_loading.setText("LOADING DATA"))
        QtCore.QTimer.singleShot(
            1000, lambda: self.ui.label_loading.setText("LOADING ICON"))
        QtCore.QTimer.singleShot(
            1500, lambda: self.ui.label_loading.setText("LOADING TRAIN"))
        QtCore.QTimer.singleShot(
            2000, lambda: self.ui.label_loading.setText("LOADING TEST"))

        self.show()

    def progress(self):
        global counter
        # Set value to progress bar
        self.ui.progressBar.setValue(counter)

        # Close splash screen & open app
        if counter > 100:
            # Stop timer
            self.timer.stop()
            # Show main window
            self.main = MainWindow()
            self.main.show()
            # Close splash screen
            self.close()
        # Increase counter
        counter += 1


app = QtWidgets.QApplication(sys.argv)
window = SplashScreen()
app.exec_()
