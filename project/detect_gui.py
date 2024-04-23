# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'detect_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_main(object):
    def setupUi(self, main):
        main.setObjectName("main")
        main.resize(1099, 830)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main.sizePolicy().hasHeightForWidth())
        main.setSizePolicy(sizePolicy)
        main.setMinimumSize(QtCore.QSize(1099, 830))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(main)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.cam_detect_box = QtWidgets.QHBoxLayout()
        self.cam_detect_box.setObjectName("cam_detect_box")
        self.cam_box = QtWidgets.QScrollArea(main)
        self.cam_box.setAutoFillBackground(False)
        self.cam_box.setLineWidth(1)
        self.cam_box.setWidgetResizable(True)
        self.cam_box.setObjectName("cam_box")
        self.cam = QtWidgets.QWidget()
        self.cam.setGeometry(QtCore.QRect(0, 0, 535, 480))
        self.cam.setObjectName("cam")
        self.cam_video = QtWidgets.QLabel(self.cam)
        self.cam_video.setGeometry(QtCore.QRect(0, 0, 537, 480))
        self.cam_video.setMinimumSize(QtCore.QSize(535, 480))
        self.cam_video.setMaximumSize(QtCore.QSize(537, 482))
        self.cam_video.setText("")
        self.cam_video.setObjectName("cam_video")
        self.cam_box.setWidget(self.cam)
        self.cam_detect_box.addWidget(self.cam_box)
        self.cam_control_box = QtWidgets.QGroupBox(main)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(15)
        self.cam_control_box.setFont(font)
        self.cam_control_box.setObjectName("cam_control_box")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.cam_control_box)
        self.verticalLayout.setObjectName("verticalLayout")
        self.num_chose_box = QtWidgets.QHBoxLayout()
        self.num_chose_box.setContentsMargins(10, -1, 10, -1)
        self.num_chose_box.setObjectName("num_chose_box")
        self.num_chose_label = QtWidgets.QLabel(self.cam_control_box)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.num_chose_label.setFont(font)
        self.num_chose_label.setObjectName("num_chose_label")
        self.num_chose_box.addWidget(self.num_chose_label)
        self.num_chose_control = QtWidgets.QComboBox(self.cam_control_box)
        self.num_chose_control.setObjectName("num_chose_control")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_control.addItem("")
        self.num_chose_box.addWidget(self.num_chose_control)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.num_chose_box.addItem(spacerItem)
        self.verticalLayout.addLayout(self.num_chose_box)
        self.x_position_box = QtWidgets.QHBoxLayout()
        self.x_position_box.setObjectName("x_position_box")
        self.x_position_label = QtWidgets.QLabel(self.cam_control_box)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.x_position_label.setFont(font)
        self.x_position_label.setObjectName("x_position_label")
        self.x_position_box.addWidget(self.x_position_label)
        self.x_position_display = QtWidgets.QTextBrowser(self.cam_control_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x_position_display.sizePolicy().hasHeightForWidth())
        self.x_position_display.setSizePolicy(sizePolicy)
        self.x_position_display.setMinimumSize(QtCore.QSize(0, 0))
        self.x_position_display.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.x_position_display.setFont(font)
        self.x_position_display.setObjectName("x_position_display")
        self.x_position_box.addWidget(self.x_position_display)
        self.x_position_box.setStretch(1, 8)
        self.verticalLayout.addLayout(self.x_position_box)
        self.y_position_box = QtWidgets.QHBoxLayout()
        self.y_position_box.setObjectName("y_position_box")
        self.y_position_label = QtWidgets.QLabel(self.cam_control_box)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.y_position_label.setFont(font)
        self.y_position_label.setObjectName("y_position_label")
        self.y_position_box.addWidget(self.y_position_label)
        self.y_position_display = QtWidgets.QTextBrowser(self.cam_control_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.y_position_display.sizePolicy().hasHeightForWidth())
        self.y_position_display.setSizePolicy(sizePolicy)
        self.y_position_display.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.y_position_display.setFont(font)
        self.y_position_display.setObjectName("y_position_display")
        self.y_position_box.addWidget(self.y_position_display)
        self.y_position_box.setStretch(0, 2)
        self.y_position_box.setStretch(1, 8)
        self.verticalLayout.addLayout(self.y_position_box)
        self.degree_box = QtWidgets.QHBoxLayout()
        self.degree_box.setObjectName("degree_box")
        self.degree_label = QtWidgets.QLabel(self.cam_control_box)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.degree_label.setFont(font)
        self.degree_label.setObjectName("degree_label")
        self.degree_box.addWidget(self.degree_label)
        self.degree_display = QtWidgets.QTextBrowser(self.cam_control_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.degree_display.sizePolicy().hasHeightForWidth())
        self.degree_display.setSizePolicy(sizePolicy)
        self.degree_display.setMaximumSize(QtCore.QSize(350, 60))
        self.degree_display.setObjectName("degree_display")
        self.degree_box.addWidget(self.degree_display)
        self.degree_box.setStretch(1, 4)
        self.verticalLayout.addLayout(self.degree_box)
        self.direction_display = QtWidgets.QTextBrowser(self.cam_control_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.direction_display.sizePolicy().hasHeightForWidth())
        self.direction_display.setSizePolicy(sizePolicy)
        self.direction_display.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.direction_display.setFont(font)
        self.direction_display.setObjectName("direction_display")
        self.verticalLayout.addWidget(self.direction_display, 0, QtCore.Qt.AlignHCenter)
        self.cam_detect_box.addWidget(self.cam_control_box)
        self.verticalLayout_2.addLayout(self.cam_detect_box)
        self.audio_detect_box = QtWidgets.QHBoxLayout()
        self.audio_detect_box.setObjectName("audio_detect_box")
        self.audio_display_box = QtWidgets.QScrollArea(main)
        self.audio_display_box.setWidgetResizable(True)
        self.audio_display_box.setObjectName("audio_display_box")
        self.audio_display = QtWidgets.QWidget()
        self.audio_display.setGeometry(QtCore.QRect(0, 0, 696, 318))
        self.audio_display.setObjectName("audio_display")
        self.audio_label = QtWidgets.QLabel(self.audio_display)
        self.audio_label.setGeometry(QtCore.QRect(0, 0, 718, 318))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.audio_label.sizePolicy().hasHeightForWidth())
        self.audio_label.setSizePolicy(sizePolicy)
        self.audio_label.setText("")
        self.audio_label.setObjectName("audio_label")
        self.audio_display_box.setWidget(self.audio_display)
        self.audio_detect_box.addWidget(self.audio_display_box)
        self.groupBox = QtWidgets.QGroupBox(main)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(15)
        font.setKerning(False)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.control_box = QtWidgets.QHBoxLayout()
        self.control_box.setObjectName("control_box")
        self.record_button = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.record_button.sizePolicy().hasHeightForWidth())
        self.record_button.setSizePolicy(sizePolicy)
        self.record_button.setObjectName("record_button")
        self.control_box.addWidget(self.record_button)
        self.time_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.time_label.sizePolicy().hasHeightForWidth())
        self.time_label.setSizePolicy(sizePolicy)
        self.time_label.setObjectName("time_label")
        self.control_box.addWidget(self.time_label)
        self.time_control = QtWidgets.QSpinBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.time_control.sizePolicy().hasHeightForWidth())
        self.time_control.setSizePolicy(sizePolicy)
        self.time_control.setProperty("value", 7)
        self.time_control.setObjectName("time_control")
        self.control_box.addWidget(self.time_control)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.control_box.addWidget(self.label_2)
        self.threshold_control = QtWidgets.QSpinBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.threshold_control.sizePolicy().hasHeightForWidth())
        self.threshold_control.setSizePolicy(sizePolicy)
        self.threshold_control.setProperty("value", 50)
        self.threshold_control.setObjectName("threshold_control")
        self.control_box.addWidget(self.threshold_control)
        self.verticalLayout_3.addLayout(self.control_box)
        self.single_box = QtWidgets.QHBoxLayout()
        self.single_box.setContentsMargins(-1, 10, -1, 10)
        self.single_box.setObjectName("single_box")
        self.single_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.single_label.sizePolicy().hasHeightForWidth())
        self.single_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(15)
        self.single_label.setFont(font)
        self.single_label.setObjectName("single_label")
        self.single_box.addWidget(self.single_label)
        self.single_display = QtWidgets.QTextBrowser(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.single_display.sizePolicy().hasHeightForWidth())
        self.single_display.setSizePolicy(sizePolicy)
        self.single_display.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        font.setKerning(False)
        self.single_display.setFont(font)
        self.single_display.setAcceptDrops(True)
        self.single_display.setAcceptRichText(True)
        self.single_display.setObjectName("single_display")
        self.single_box.addWidget(self.single_display)
        self.verticalLayout_3.addLayout(self.single_box)
        self.up_box = QtWidgets.QHBoxLayout()
        self.up_box.setContentsMargins(-1, 10, -1, 10)
        self.up_box.setObjectName("up_box")
        self.up_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_label.sizePolicy().hasHeightForWidth())
        self.up_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(15)
        self.up_label.setFont(font)
        self.up_label.setObjectName("up_label")
        self.up_box.addWidget(self.up_label)
        self.up_display = QtWidgets.QTextBrowser(self.groupBox)
        self.up_display.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.up_display.setFont(font)
        self.up_display.setObjectName("up_display")
        self.up_box.addWidget(self.up_display)
        self.verticalLayout_3.addLayout(self.up_box)
        self.down_box = QtWidgets.QHBoxLayout()
        self.down_box.setContentsMargins(-1, 10, -1, 10)
        self.down_box.setObjectName("down_box")
        self.down_label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_label.sizePolicy().hasHeightForWidth())
        self.down_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(15)
        self.down_label.setFont(font)
        self.down_label.setObjectName("down_label")
        self.down_box.addWidget(self.down_label)
        self.down_display = QtWidgets.QTextBrowser(self.groupBox)
        self.down_display.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.down_display.setFont(font)
        self.down_display.setObjectName("down_display")
        self.down_box.addWidget(self.down_display)
        self.verticalLayout_3.addLayout(self.down_box)
        self.result_display = QtWidgets.QTextBrowser(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result_display.sizePolicy().hasHeightForWidth())
        self.result_display.setSizePolicy(sizePolicy)
        self.result_display.setMinimumSize(QtCore.QSize(0, 0))
        self.result_display.setMaximumSize(QtCore.QSize(200, 50))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.result_display.setFont(font)
        self.result_display.setObjectName("result_display")
        self.verticalLayout_3.addWidget(self.result_display, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 2)
        self.verticalLayout_3.setStretch(2, 2)
        self.verticalLayout_3.setStretch(3, 2)
        self.verticalLayout_3.setStretch(4, 2)
        self.audio_detect_box.addWidget(self.groupBox)
        self.audio_detect_box.setStretch(0, 7)
        self.verticalLayout_2.addLayout(self.audio_detect_box)
        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 4)

        self.retranslateUi(main)
        QtCore.QMetaObject.connectSlotsByName(main)

    def retranslateUi(self, main):
        _translate = QtCore.QCoreApplication.translate
        main.setWindowTitle(_translate("main", "Form"))
        self.cam_control_box.setTitle(_translate("main", "camera control"))
        self.num_chose_label.setText(_translate("main", "number to detect:"))
        self.num_chose_control.setItemText(0, _translate("main", "1"))
        self.num_chose_control.setItemText(1, _translate("main", "2"))
        self.num_chose_control.setItemText(2, _translate("main", "3"))
        self.num_chose_control.setItemText(3, _translate("main", "4"))
        self.num_chose_control.setItemText(4, _translate("main", "5"))
        self.num_chose_control.setItemText(5, _translate("main", "6"))
        self.num_chose_control.setItemText(6, _translate("main", "7"))
        self.num_chose_control.setItemText(7, _translate("main", "8"))
        self.num_chose_control.setItemText(8, _translate("main", "9"))
        self.x_position_label.setText(_translate("main", "x position:"))
        self.x_position_display.setHtml(_translate("main", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial Rounded MT Bold\'; font-size:20pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.y_position_label.setText(_translate("main", "y position:"))
        self.degree_label.setText(_translate("main", "Degree:"))
        self.direction_display.setHtml(_translate("main", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial Rounded MT Bold\'; font-size:20pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:15pt;\"><br /></p></body></html>"))
        self.groupBox.setTitle(_translate("main", "audio control"))
        self.record_button.setText(_translate("main", "Record"))
        self.time_label.setText(_translate("main", "time:"))
        self.label_2.setText(_translate("main", "threshold:"))
        self.single_label.setText(_translate("main", "single:"))
        self.single_display.setHtml(_translate("main", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial Rounded MT Bold\'; font-size:20pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:15pt;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:15pt;\"><br /></p></body></html>"))
        self.up_label.setText(_translate("main", "       up:"))
        self.down_label.setText(_translate("main", "  down:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QWidget()
    ui = Ui_main()
    ui.setupUi(main)
    main.show()
    sys.exit(app.exec_())