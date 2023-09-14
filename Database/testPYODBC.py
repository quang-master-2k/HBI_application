import pyodbc

server = 'quangsog-Inspiron-5570'
database = 'testApp'
username = 'sa'
password = '123456Qa'
connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(connection_string)

cursor = conn.cursor()
cursor.execute("SELECT * FROM bbb")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()