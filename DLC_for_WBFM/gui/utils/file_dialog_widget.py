# From:
# https://pythonspot.com/pyqt5-file-dialog/

import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog


class FileDialog(QWidget):

    def __init__(self, file_not_folder=True):
        super().__init__()
        self.file_not_folder = file_not_folder
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        if self.file_not_folder:
            self.openFileNameDialog()
        else:
            self.openFolderDialog()

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "All Files (*);;Python Files (*.py)", options=options)

    def openFolderDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName = QFileDialog.getExistingDirectory(self, "Select Directory", "")

    # def openFileNamesDialog(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
    #                                             "All Files (*);;Python Files (*.py)", options=options)
    #     if files:
    #         print(files)
    #
    # def saveFileDialog(self):
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.DontUseNativeDialog
    #     fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
    #                                               "All Files (*);;Text Files (*.txt)", options=options)
    #     if fileName:
    #         print(fileName)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileDialog()
    sys.exit(app.exec_())
