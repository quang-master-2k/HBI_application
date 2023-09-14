import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget, QLabel
from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery

def createDB():
    db = QSqlDatabase.addDatabase('QSQLITE')
    db.setDatabaseName('databaseTest.db')  # Replace 'your_database_name.db' with your database filename
    if not db.open():
        print("Unable to open database")
        sys.exit(1)
    return db

def fetchData(db):
    query = QSqlQuery('''SELECT Size, L1/AC954CCL, L2/AC954CCL, L3/AC954CCL, L4/AC954CCL, L5/AC954CCL FROM data_specs''')
    model = QSqlQueryModel()
    model.setQuery(query)
    if model.lastError().isValid():
        print("Error: " + model.lastError().text())
    return model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.layout = QVBoxLayout(self.central_widget)

        self.label = QLabel("Your QTableView:", self.central_widget)
        self.layout.addWidget(self.label)

        self.table_view = QTableView(self.central_widget)
        self.layout.addWidget(self.table_view)

        self.setCentralWidget(self.central_widget)
        self.setGeometry(100, 100, 800, 600)

        # Connect to the database and retrieve data
        db = createDB()
        model = fetchData(db)

        # Set the model for the QTableView
        self.table_view.setModel(model)
        db.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
