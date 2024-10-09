import mysql.connector
import random
from datetime import datetime, timedelta

def create_table():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="SoujuK@1799"
    )
    mycursor = mydb.cursor()
    mycursor.execute(f"CREATE DATABASE IF NOT EXISTS footfall")
    mycursor.execute(f"USE footfall")
    # Create a table if not exists
    mycursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_faces (
            id INT AUTO_INCREMENT PRIMARY KEY,
            face_id VARCHAR(255),
            date_time DATETIME,
            cropped_frame BLOB
        )
    """)
    mydb.close()

def generate_random_face_id():
    return f"face_{random.randint(1000, 9999)}"

def fill_data():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="SoujuK@1799",
        database="footfall"
    )
    mycursor = mydb.cursor()

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    current_date = start_date
    while current_date <= end_date:
        current_time = datetime(current_date.year, current_date.month, current_date.day, 0, 0)  # Start from midnight
        while current_time.day == current_date.day:
            # Insert more face_ids on Mondays and Wednesdays
            num_entries = 15 if current_time.weekday() in [0, 2] else 5  # Monday is 0 and Wednesday is 2
            for _ in range(num_entries):
                face_id = generate_random_face_id()
                sql = "INSERT INTO detected_faces (face_id, date_time, cropped_frame) VALUES (%s, %s, %s)"
                val = (face_id, current_time.strftime('%Y-%m-%d %H:%M:%S'), None)
                mycursor.execute(sql, val)

            current_time += timedelta(minutes=8)

        current_date += timedelta(days=1)

    mydb.commit()
    mydb.close()

if __name__ == "__main__":
    create_table()
    fill_data()
    print("Data insertion complete.")
