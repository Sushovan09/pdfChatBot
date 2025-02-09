from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import os

app = Flask(__name__)

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='mydb',
            user='pankaj',
            password='pankaj'
        )
        print("Connected to MySQL database")
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
    return connection

UPLOAD_FOLDER = '/home/system/ET/llmapp/books'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    # Save file path to MySQL database
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            sql_query = "INSERT INTO books (book_path) VALUES (%s)"
            cursor.execute(sql_query, (os.path.join(app.config['UPLOAD_FOLDER'], file.filename),))
            connection.commit()
            print("File path saved to MySQL database")
        except Error as e:
            print(f"Error inserting file path into MySQL database: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection closed")
    else:
        print("Failed to connect to MySQL database")

    return jsonify({'message': 'File uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)

