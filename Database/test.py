import sqlite3
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTableView
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5 import QtCore, QtGui, QtWidgets

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self._data.columns[section])
        return None


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 101, 17))
        self.label.setObjectName("label")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(115, 3, 131, 31))
        self.textEdit.setObjectName("textEdit")

        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(365, 3, 131, 31))
        self.textEdit_2.setObjectName("textEdit_2")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 10, 101, 17))
        self.label_2.setObjectName("label_2")

        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(615, 3, 131, 31))
        self.textEdit_3.setObjectName("textEdit_3")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(510, 10, 101, 17))
        self.label_3.setObjectName("label_3")

        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(10, 50, 671, 31))
        self.textEdit_4.setObjectName("textEdit_4")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(700, 50, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.open_file_dialog)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 100, 121, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.add_database)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(800, 3, 89, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.show_database)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 150, 1000, 17))
        self.label_4.setObjectName("label_4")

        self.tableView = QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(20, 200, 1000, 500))
        self.tableView.setObjectName("tableView")

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Garment Style"))
        self.label_2.setText(_translate("MainWindow", "Pattern Code"))
        self.label_3.setText(_translate("MainWindow", "Piece Name"))
        self.label_4.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "Browse"))
        self.pushButton_2.setText(_translate("MainWindow", "Add Database"))
        self.pushButton_3.setText(_translate("MainWindow", "Show"))

    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path = file_dialog.getOpenFileName(None, "Select File")[0]
        self.textEdit_4.setText(file_path)

    def add_database(self):
        file_path = self.textEdit_4.toPlainText()
        text_insert = self.get_specs_info()
        if file_path:
            try:
                conn = sqlite3.connect('databaseTest.db')  # Replace 'database.db' with your desired database name
                cursor = conn.cursor()
                wb = pd.read_excel(file_path)
                table_name = 'data_specs'  # Replace 'my_table' with your desired table name
                for index, row in wb.iterrows():
                    size = row['Size']
                    size = int(size)
                    for column_name, value in row.items():
                        if column_name != 'Size': 
                            dimension_name = column_name
                            dimension_value = value
                            text_query = text_insert.copy()
                            text_query.append(size)
                            text_query.append(dimension_name)
                            text_query.append(dimension_value)
                            sql = '''INSERT INTO data_specs (Garment_Style, Pattern_Code, Piece_Name, Size, Dimension_Name, Dimension_Value)
                            VALUES (?,?,?,?,?,?)'''

                            conn.execute(sql, text_query)
                            conn.commit()

                conn.close()
                
                print("Succesful")

            except Exception as e:
                print("Error")
        else:
            print("No file selected")
        
        conn = sqlite3.connect('databaseTest.db')  # Replace 'database.db' with your desired database name
        cursor = conn.cursor()
        wb = pd.read_excel(file_path)
        table_name = text_insert[0]+text_insert[1]+text_insert[2]  # Replace 'my_table' with your desired table name

        wb.to_sql(table_name, conn, if_exists='replace', index=False)

        conn.commit()
        conn.close()

        self.show_database()

    def get_specs_info(self):
        textEdits = [
            self.textEdit,
            self.textEdit_2,
            self.textEdit_3
        ]

        texts = []
        for textEdit in textEdits:
            text = textEdit.toPlainText()
            texts.append(text)

        return texts

    def show_database(self):
        conn = sqlite3.connect('databaseTest.db')  # Replace 'database.db' with your desired database name
        cursor = conn.cursor()
        text_insert = self.get_specs_info()
        table_name = text_insert[0]+text_insert[1]+text_insert[2]  # Replace 'my_table' with your desired table name
        # Display the data table
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        headers = list(df.head(0))
        model = PandasModel(df)
        self.tableView.setModel(model)
        conn.close()
        self.label_4.setText('Data: ' + text_insert[0]+text_insert[1]+text_insert[2])
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        self.label_4.setFont(font)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
