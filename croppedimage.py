import mysql.connector
from mysql.connector import Error
import cv2
import numpy as np
from PIL import Image

def retrieve_cropped_frames():
    try:
        # Connect to MySQL database
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )

        # Create a cursor object to execute queries
        mycursor = mydb.cursor()

        # Select cropped_frame from detected_faces table
        mycursor.execute("SELECT cropped_frame FROM detected_faces")

        # Fetch all rows
        rows = mycursor.fetchall()

        # Process each row
        for row in rows:
            # Assuming cropped_frame is stored as bytes in the database
            cropped_frame_bytes = row[0]

            # Check if cropped_frame_bytes is None
            if cropped_frame_bytes is None:
                print("No image data found in this row.")
                continue

            # Convert bytes back to image array
            nparr = np.frombuffer(cropped_frame_bytes, np.uint8)
            if len(nparr) == 0:
                print("Empty or invalid image data retrieved.")
                continue

            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV image array

            # Check if image array is valid
            if img_np is None:
                print("Failed to decode image data.")
                continue

            # Convert to RGB format for PIL
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Display or process the image (example: display using PIL)
            img_pil.show()  # Opens image in default viewer

        # Close cursor and database connection
        mycursor.close()
        mydb.close()

    except Error as e:
        print(f"Error retrieving data from database: {e}")

# Call the function to retrieve cropped frames
retrieve_cropped_frames()
