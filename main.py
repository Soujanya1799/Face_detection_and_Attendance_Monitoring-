import argparse
import datetime
import json
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, jsonify,request
from flask_socketio import SocketIO
import gevent          
from gevent import pywsgi                            #new
from geventwebsocket.handler import WebSocketHandler #new
from flask_sockets import Sockets    
from flask_cors import CORS
import base64
import mysql.connector
import time
from mysql.connector import Error
import camera
from mysql.connector.pooling import MySQLConnectionPool


app = Flask(__name__)

# socketio = SocketIO(app, cors_allowed_origins="*")
# cors = CORS(app, resources={r"/": {"origins": ""}})
sockets = Sockets(app)  #Sockets are used for connecting two nodes on network.  
cors = CORS(app, resources={r"/*": {"origins": "*"}}) #CORS is used for sending data from one website to another,which sometimes get prohibited. 
                                                      #origin

# Initialize connection pool
db_pool = MySQLConnectionPool(
    # pool_name="local",
    # pool_size=5,
    host="localhost",
    user="root",
    password="SoujuK@1799",
    database="footfall"
)


def configure_cameras(args, frame_dict):
    camera_threads = []
    camera_config = camera.load_camera_config(args.cameras_json)

    for _ , (cam_serial_num, rtsp_path) in enumerate(camera_config.items()):
        thread = camera.CameraThread(cam_serial_num, rtsp_path, frame_dict, args.fps)
        camera_threads.append(thread)
        thread.start()
    return camera_threads

def load_model(model):
    return YOLO(model)

def plot_roi(latest_frame, polygon, linepoints):
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(latest_frame, [pts], isClosed=True, color=(0,0,255), thickness=3)
    cv2.line(latest_frame, linepoints[0], linepoints[1], color=(0, 255, 255), thickness=2)

def plot_bbox_withid(latest_frame,box,id):
    str_id = str(int(id.item()))
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(latest_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(latest_frame, str_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

def inside_or_outside(centroid,linepoints):
    if (centroid[1]<linepoints[0][1]) or (centroid[1] < linepoints[1][1]):
        return "outside"
    return "inside"

def centroid(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    centroid = [(x1+x2)//2, (y1+y2)//2]
    return centroid


def footfall_condition(centroid,id,crossed_ids,linepoints,polygon):
    str_id = str(int(id.item()))
    if (centroid[0] > polygon[0][0]) and (centroid[0]< polygon[1][0]):
        if (centroid[1]<linepoints[0][1]) or (centroid[1] < linepoints[1][1]):  # If the centroid is above the line
                if str_id not in crossed_ids:
                    crossed_ids[str_id] = True
                return False, crossed_ids

        else:  # If the centroid is below the line
            if str_id in crossed_ids:
                if crossed_ids[str_id]:
                    # del crossed_ids[str_id]
                    crossed_ids[str_id] = False
                    return True, crossed_ids
            return False, crossed_ids
    else:
        return False, crossed_ids
#save_detected_face_data_in_db



def crop_frame(latest_frame, box):
    padding=20
    # Extract box coordinates
    x1, y1, x2, y2 = box.xyxy.squeeze().tolist()

    # Ensure coordinates are within bounds
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(latest_frame.shape[1], x2 + padding)
    y2 = min(latest_frame.shape[0], y2 + padding)

    # Crop the frame using the adjusted box coordinates
    cropped_frame = latest_frame[y1:y2, x1:x2]

    return cropped_frame



def save_data_in_database(self,latest_frame, box, face_id, date_time,entry_count):
    cropped_frame_bytes = None
    try:
        # start_time = time.time() * 1000  # Start time in milliseconds

        # Crop the frame using the box coordinates
        cropped_frame = crop_frame(latest_frame, box)
        
        # Convert cropped frame to bytes
        _, cropped_frame_bytes = cv2.imencode('.jpg', cropped_frame)
        cropped_frame_bytes = cropped_frame_bytes.tobytes()
        # Store cropped_frame_bytes in PeopleCounter instance
        self.cropped_frame_bytes = cropped_frame_bytes
        
        # Get a connection from the pool
        connection = db_pool.get_connection()


        # Create a cursor object to execute queries
        with connection.cursor() as mycursor:
            # Create table if not exists
            mycursor.execute("CREATE TABLE IF NOT EXISTS detected_faces (id INT AUTO_INCREMENT PRIMARY KEY, face_id VARCHAR(255), date_time DATETIME, cropped_frame LONGBLOB, entry_count INT)")

            # Insert data into the table
            sql = "INSERT INTO detected_faces (face_id, date_time, cropped_frame, entry_count) VALUES (%s, %s, %s, %s)"
            val = (face_id, date_time, cropped_frame_bytes, entry_count)
            mycursor.execute(sql, val)

        # Commit changes
        connection.commit()

        # end_time = time.time() * 1000  # End time in milliseconds
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time:.2f} ms")

    except Error as e:
        print(f"Error saving data to database: {e}")

    finally:
        # Release the connection back to the pool
        if connection:
            connection.close()


# def get_roi(roi_file):
#     with open(roi_file,"r") as f:
#         roi_file_content = f.read()
#         polygon = json.loads(roi_file_content)
#         point1 = polygon["polygon"]["point1"]
#         point2 = polygon["polygon"]["point2"]
#         point3 = polygon["polygon"]["point3"]
#         point4 = polygon["polygon"]["point4"]
#         polygon_points = [point1, point2, point3, point4]

#         line_point1 = polygon["line"]["point1"]
#         line_point2 = polygon["line"]["point2"]
#         line_points = [line_point1,line_point2]

#         return polygon_points, line_points



            

class PeopleCounter():
    def __init__(self, args) -> None:
        self.timestamp = ''
        self.entry_count = self.load_entry_count()  # Initialize entry_count using method to load from database
        
        self.video_path = args.video
        self.video = None
        
        # Initialize video capture if a video path is provided
        if self.video_path:
            self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise ValueError(f"Error: Failed to open video file {self.video_path}")
        else:
            raise ValueError("Error: No video file path provided")

        self.visible_inside = 0
        self.visible_outside = 0
        self.crossed_ids = {}
        self.frame_dict = {}
        self.frame_shape = (1280, 720)
        self.error_image = np.zeros((self.frame_shape[0], self.frame_shape[1], 3), dtype="uint8")
        self.latest_frame = np.ones((self.frame_shape[0], self.frame_shape[1], 3), dtype="uint8")
        
        # Configure the model and ROI
        self.model = load_model(args.model)
        # self.polygon, self.linepoints = get_roi(args.roi_file)
        
        self.cropped_frame_bytes = None  # Initialize cropped_frame_bytes
        self.last_reset_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  # Initialize last reset time to today at midnight

    
    def get_roi(self):
        try:
            # Connect to your MySQL database
            db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="SoujuK@1799",
                database="footfall"
            )
            cursor = db.cursor(dictionary=True)
            
    
            # Create table if not exists
            create_table_query = """
                CREATE TABLE IF NOT EXISTS roi_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    polygon_coordinates TEXT,
                    polygon_coordinates_original TEXT,
                    h_line_coordinates VARCHAR(255),
                    original_h_line_coordinates VARCHAR(255),
                    datetime_insertion DATETIME
                )
            """
            cursor.execute(create_table_query)

            # Assuming you have stored ROI data in a table named 'roi_data'
            select_query = "SELECT * FROM roi_data ORDER BY id DESC LIMIT 1"
            cursor.execute(select_query)
            roi_data = cursor.fetchone()
            

            if not roi_data:
            # Define default ROI data in JSON format
                default_data = {
                    "polygon": {
                        "point1": [300, 50],
                        "point2": [700, 50],
                        "point3": [700, 650],
                        "point4": [300, 650]
                    },
                    "line": {
                        "point1": [300, 300],
                        "point2": [700, 300]
                    }
                }

                # Print the default data for reference or logging
                print("No ROI data found in the database. Using default data.")
                print("Default ROI data:", default_data)

                # Return default data as Python dictionaries
                return (
                    [
                        default_data["polygon"]["point1"],
                        default_data["polygon"]["point2"],
                        default_data["polygon"]["point3"],
                        default_data["polygon"]["point4"]
                    ],
                    [
                        default_data["line"]["point1"],
                        default_data["line"]["point2"]
                    ]
                )

            # Extract polygon coordinates from JSON string
            polygon_coordinates = json.loads(roi_data['polygon_coordinates'])
            polygon_points = [
                polygon_coordinates[0][0],  # Point 1 [x, y]
                polygon_coordinates[0][1],  # Point 2 [x, y]
                polygon_coordinates[0][2],  # Point 3 [x, y]
                polygon_coordinates[0][3]   # Point 4 [x, y]
            ]

           # Extract and parse line coordinates
            line_coordinates_str = roi_data['h_line_coordinates']
            line_coordinates = list(map(int, line_coordinates_str.split(',')))
            line_points = [
                [line_coordinates[0], line_coordinates[2]],  # Point 1 [x1, x2]
                [line_coordinates[1], line_coordinates[3]]   # Point 2 [y1, y2]
            ]

            print("polygon_points:", polygon_points)
            print("line_points:", line_points)
            return polygon_points, line_points

        except mysql.connector.Error as db_err:
            print(f"Database Error: {db_err}")
            return None, None

        finally:
            if cursor:
                cursor.close()
            if db:
                db.close()
    
    
    
    def load_entry_count(self):
        try:
            # Connect to the database (use your connection method)
            connection = db_pool.get_connection()

            # Check if there's a record for today
            today = datetime.date.today()
            sql = "SELECT entry_count FROM detected_faces WHERE DATE(date_time) = %s ORDER BY date_time DESC LIMIT 1"
            with connection.cursor() as mycursor:
                mycursor.execute(sql, (today,))
                result = mycursor.fetchone()

            if result:
                return result[0]  # Return entry_count from the database
            else:
                return 0  # If no record found for today, return 0

        except Error as e:
            print(f"Error loading entry count from database: {e}")
            return 0  # Return 0 on error

        finally:
            if connection:
                connection.close()
    
    
    def run(self):
        # Check if it's time to reset entry_count
        current_time = datetime.datetime.now()
        if current_time < self.last_reset_time:  # If current time is before midnight, update last reset time to yesterday
            self.last_reset_time = self.last_reset_time.replace(day=current_time.day - 1)

        # Check if it's a new day and reset entry_count if needed
        if current_time.day != self.last_reset_time.day:
            self.entry_count = 0
            self.crossed_ids = {}
            self.last_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # Update last reset time to today at midnight

        self.visible_inside = 0
        self.visible_outside = 0
        
        for thread in self.camera_threads:
            latest_frame = self.frame_dict.get(thread.cam_serial_num, self.error_image)
            self.latest_frame_without_roi = latest_frame
            self.latest_frame_roi = latest_frame.copy()
            detections = self.model.track(self.latest_frame_roi, persist=True, classes=0, conf=0.6,verbose=False)

            for objects in detections:
                boxes = objects.boxes
                for box in boxes:
                    try:
                        id = box.id[0]
                    except TypeError:
                        continue

                    box_centroid = centroid(box)

                    side = inside_or_outside(box_centroid, self.linepoints)
                    if side == "inside":
                        self.visible_inside += 1
                    else:
                        self.visible_outside += 1

                    count_condition, self.crossed_ids = footfall_condition(box_centroid,id, self.crossed_ids,self.linepoints,self.polygon)
                    # print("count_condition::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::",count_condition)
                    if count_condition:
                        self.entry_count += 1
                        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current timestamp
                        save_data_in_database(self,self.latest_frame_without_roi, box, id.item(), self.timestamp,self.entry_count)
                        # save_data_in_database(latest_frame, id.item(), self.timestamp)
                    else:
                        pass

                    plot_bbox_withid(self.latest_frame_roi, box, id)
            plot_roi(self.latest_frame_roi, self.polygon, self.linepoints)
        # self.latest_frame_roi = latest_frame

        # print("self.entry_count::::::",self.entry_count)
        # print("self.visible_inside::::",self.visible_inside)
        # print("self.visible_outside::::",self.visible_outside)
        

@sockets.route('/footfall_feed')
def stream_video(ws):
    print("footfall_feed websocket handshaked")
    with app.app_context():
        try:
            counter_obj = PeopleCounter(arguments)  # Create an instance of PeopleCounter
            while not ws.closed:
                counter_obj.run()
                
                frame = np.squeeze(counter_obj.latest_frame_roi)
                latest_frame_without_roi = np.squeeze(counter_obj.latest_frame_without_roi)
                if frame is not None and latest_frame_without_roi is not None:
                    height, width = frame.shape[:2]  # Extract height and width
                    frame = cv2.resize(frame, (width, height))  # Resize with (width, height)
                    latest_frame_without_roi = cv2.resize(latest_frame_without_roi, (width, height))  # Resize with (width, height)
                    # cv2.imshow('test', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                    
                    _, buffer_without_roi = cv2.imencode('.jpg', latest_frame_without_roi)
                    frame_bytes_without_roi = buffer_without_roi.tobytes()
                    frame_base64_without_roi = base64.b64encode(frame_bytes_without_roi).decode('utf-8')

                    # Handle NoneType error for cropped_frame_bytes
                    if counter_obj.cropped_frame_bytes is not None:
                        cropped_frame = base64.b64encode(counter_obj.cropped_frame_bytes).decode('utf-8')
                    else:
                        cropped_frame = None  # or handle appropriately for your application

                    # print("counter_obj.entry_count",counter_obj.entry_count)
                    
                    data = {
                        'image': frame_base64,
                        'image_without_roi':frame_base64_without_roi, 
                        'cropped_frame':cropped_frame, 
                        'visible_inside': counter_obj.visible_inside,
                        'visible_outside': counter_obj.visible_outside,
                        'entry_count': counter_obj.entry_count,
                        'timestamp': datetime.datetime.now().strftime("%I:%M %p")
                        # 'timestamp': datetime.datetime.now().strftime("%I:%M:%S %p")
                    }
                    ws.send(json.dumps(data))
                gevent.sleep(0.01)
        except Exception as e:
            print(f"Error streaming video: {e}")
            if not ws.closed:
                ws.send(json.dumps({'error': str(e)}))             
        finally:
            print("Client disconnected")

# API to get present day's hourly count
@app.route('/api/present-day', methods=['GET'])
def get_present_day():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)
        query = """
            SELECT HOUR(date_time) AS hour, COUNT(face_id) AS count
            FROM detected_faces
            WHERE DATE(date_time) = CURDATE()
            GROUP BY HOUR(date_time)
            ORDER BY hour;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        print(".....................................................................................................................................................................")
        print("present day Data........................................",results)
        print("......................................................................................................................................................................")
        cursor.close()
        db.close()
        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# API to get last 30 days' data
@app.route('/api/last-30-days', methods=['GET'])
def get_last_30_days():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)
        
        
        
        shift_query = """
            SELECT *
            FROM shift_data
        """
        cursor.execute(shift_query)
        shift_data = cursor.fetchone()
        
        shift1Name = shift_data['shift_one_name']
        shift2Name = shift_data['shift_two_name']
        shift3Name = shift_data['shift_three_name']
        
        
       
        query = """
            SELECT 
                DATE(date_time) AS date,
                COUNT(CASE WHEN HOUR(date_time) >= '{}' AND HOUR(date_time) < '{}' THEN id END) AS shift1_count,
                COUNT(CASE WHEN HOUR(date_time) >= '{}' AND HOUR(date_time) < '{}' THEN id END) AS shift2_count,
                COUNT(CASE WHEN HOUR(date_time) >= '{}' AND HOUR(date_time) < '{}' THEN id END) AS shift3_count
            FROM detected_faces
            WHERE date_time >= CURDATE() - INTERVAL 30 DAY
            GROUP BY DATE(date_time)
            ORDER BY date;
        """.format(shift_data['shift_one_start'], shift_data['shift_one_end'],
                   shift_data['shift_two_start'], shift_data['shift_two_end'],
                   shift_data['shift_three_start'], shift_data['shift_three_end'])
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Prepare response data including shift names and results
        response_data = {
            'shift1Name': shift1Name,
            'shift2Name': shift2Name,
            'shift3Name': shift3Name,
            'results': results
        }
        
        print(".....................................................................................................................................................................")
        print("30 days Data........................................",response_data)
        print("......................................................................................................................................................................")
        cursor.close()
        db.close()
        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# API to get last 30 days' shift data
@app.route('/api/shift-data-last-30days', methods=['GET'])
def get_shift_data_Last_30days():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)
        
        # Query to retrieve shift timings from shift_data table
        shift_query = """
            SELECT *
            FROM shift_data
        """
        cursor.execute(shift_query)
        shift_data = cursor.fetchone()  # Assuming there's only one row in shift_data for simplicity
                
        query = """
            SELECT
                DAYNAME(date_time) AS day,
                COUNT(CASE WHEN TIME(date_time)  >= '{}' AND TIME(date_time) < '{}' THEN face_id END) AS '{}',
                COUNT(CASE WHEN TIME(date_time)  >= '{}' AND TIME(date_time) < '{}' THEN face_id END) AS '{}',
                COUNT(CASE WHEN TIME(date_time)  >= '{}' AND TIME(date_time) < '{}' THEN face_id END) AS '{}'
            FROM detected_faces
            WHERE date_time >= CURDATE() - INTERVAL 30 DAY
            GROUP BY DAYNAME(date_time)
        """.format(shift_data['shift_one_start'], shift_data['shift_one_end'],shift_data['shift_one_name'],
                   shift_data['shift_two_start'], shift_data['shift_two_end'],shift_data['shift_two_name'],
                   shift_data['shift_three_start'], shift_data['shift_three_end'], shift_data['shift_three_name'])

        
        
        cursor.execute(query)
        results = cursor.fetchall()
        print(".....................................................................................................................................................................")
        print("Shiftwise Data........................................",results)
        print("......................................................................................................................................................................")
        cursor.close()
        db.close()

        # Custom sorting based on the day of the week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sorted_results = sorted(results, key=lambda x: day_order.index(x['day']))

        return jsonify(sorted_results)
    except mysql.connector.Error as db_err:
        print(f"Database Error: {db_err}")
        return jsonify({"error": "Database Error", "message": str(db_err)}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500


@app.route('/api/shift-data', methods=['GET'])
def get_shift_data():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)
        
        # Query to retrieve shift timings from shift_data table
        shift_query = """
            SELECT *
            FROM shift_data
        """
        cursor.execute(shift_query)
        shift_data = cursor.fetchone()  # Assuming there's only one row in shift_data for simplicity
        print(".....................................................................................................................................................................")
        print("shift Timings data........................................",shift_data)
        print("......................................................................................................................................................................")
        cursor.close()
        db.close()
        
        return jsonify(shift_data)
    except mysql.connector.Error as db_err:
        print(f"Database Error: {db_err}")
        return jsonify({"error": "Database Error", "message": str(db_err)}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500
        

@app.route('/api/save-roi-data', methods=['POST'])
def save_roi_data():
    try:
        data = request.get_json()

        # Extract data from JSON
        horizontal = data.get('horizontal', [])
        horizontal_original = data.get('horizontal_original', [])
        polygon = data.get('polygon', [])
        polygon_original = data.get('polygon_original', [])

        # Assuming 'horizontal' and 'polygon' are arrays with numerical values
        x1 = horizontal[0]
        x2 = horizontal[1]
        y1 = horizontal[2]
        y2 = horizontal[3]
        o_x1 = horizontal_original[0]
        o_x2 = horizontal_original[1]
        o_y1 = horizontal_original[2]
        o_y2 = horizontal_original[3]

        # Convert 'polygon' array of arrays to a string for storage
        polygon_coordinates = str(polygon)
        polygon_coordinates_original = str(polygon_original)

        # Assuming you also receive datetime information
        datetime_insertion = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Connect to MySQL database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)

        # Create table if not exists
        create_table_query = """
            CREATE TABLE IF NOT EXISTS roi_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                polygon_coordinates TEXT,
                polygon_coordinates_original TEXT,
                h_line_coordinates VARCHAR(255),
                original_h_line_coordinates VARCHAR(255),
                datetime_insertion DATETIME
            )
        """
        cursor.execute(create_table_query)

        # Insert data into MySQL table
        insert_query = "INSERT INTO roi_data (polygon_coordinates, polygon_coordinates_original, h_line_coordinates, original_h_line_coordinates,datetime_insertion) VALUES (%s,%s, %s, %s, %s)"
        cursor.execute(insert_query, (polygon_coordinates, polygon_coordinates_original,f"{x1},{x2},{y1},{y2}", f"{o_x1},{o_x2},{o_y1},{o_y2}", datetime_insertion))
        db.commit()

        cursor.close()
        db.close()
        
        people_counter.polygon, people_counter.linepoints=people_counter.get_roi()
        print("people_counter.polygon",people_counter.polygon)
        return jsonify({'message': 'Data saved successfully'}), 200

    except mysql.connector.Error as db_err:
        print(f"Database Error: {db_err}")
        return jsonify({"error": "Database Error", "message": str(db_err)}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500


@app.route('/api/get-roi-data', methods=['GET'])
def get_roi_data():
    try:
        # Connect to MySQL database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)

        # Query to fetch the latest ROI data
        select_query = "SELECT * FROM roi_data ORDER BY id DESC LIMIT 1"
        cursor.execute(select_query)
        roi_data = cursor.fetchone()

        if not roi_data:
            # Define default ROI data in JSON format
            default_data = {
                "polygon": {
                    "point1": [300, 50],
                    "point2": [700, 50],
                    "point3": [700, 650],
                    "point4": [300, 650]
                },
                "line": {
                    "point1": [300, 300],
                    "point2": [700, 300]
                }
            }
            
            # Print the default data for reference or logging
            print("No ROI data found in the database. Using default data.")
            print("Default ROI data:", default_data)
            
            # Return default data as JSON response
            return jsonify(default_data), 200

        # Extract polygon coordinates from JSON string
        polygon_coordinates_str = roi_data['polygon_coordinates_original']
        # polygon_coordinates = json.loads(roi_data['polygon_coordinates_original'])
        polygon_coordinates = json.loads(polygon_coordinates_str)

        # Extract and parse line coordinates
        line_coordinates_str = roi_data['original_h_line_coordinates']
        line_coordinates = list(map(int, line_coordinates_str.split(',')))

       
        response = {
            "polygon": {
                "point1": polygon_coordinates[0][0],
                "point2": polygon_coordinates[0][1],
                "point3": polygon_coordinates[0][2],
                "point4": polygon_coordinates[0][3]
            },
            "line": {
                "point1": [line_coordinates[0], line_coordinates[2]],
                "point2": [line_coordinates[1], line_coordinates[3]]
            }
        }

        return jsonify(response), 200

    except mysql.connector.Error as db_err:
        print(f"Database Error: {db_err}")
        return jsonify({"error": "Database Error", "message": str(db_err)}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


@app.route('/api/saveShiftTimings', methods=['POST'])
def save_ShiftTimings():
    try:
        data = request.get_json()
        print("datasaving shift.",data)
        # Extract data from JSON
        shift1 = data.get('shift1', {})
        shift2 = data.get('shift2', {})
        shift3 = data.get('shift3', {})

        shift_one_name = shift1.get('name')
        shift_one_start = shift1.get('start')
        shift_one_end = shift1.get('end')

        shift_two_name = shift2.get('name')
        shift_two_start = shift2.get('start')
        shift_two_end = shift2.get('end')

        shift_three_name = shift3.get('name')
        shift_three_start = shift3.get('start')
        shift_three_end = shift3.get('end')
        # Connect to MySQL database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="SoujuK@1799",
            database="footfall"
        )
        cursor = db.cursor(dictionary=True)

        # Create table if not exists
        create_table_query = """
            CREATE TABLE IF NOT EXISTS shift_data (
                shift_one_name VARCHAR(255),
                shift_one_start VARCHAR(255),
                shift_one_end VARCHAR(255),
                shift_two_name VARCHAR(255),
                shift_two_start VARCHAR(255),
                shift_two_end VARCHAR(255),
                shift_three_name VARCHAR(255),
                shift_three_start VARCHAR(255),
                shift_three_end VARCHAR(255)
            )
        """
        cursor.execute(create_table_query)

        # Check if data already exists
        select_query = "SELECT * FROM shift_data"
        cursor.execute(select_query)
        existing_data = cursor.fetchone()

        if existing_data:
            # Update existing row
            
            update_query = """
                UPDATE shift_data
                SET shift_one_name=%s, shift_one_start=%s, shift_one_end=%s,
                    shift_two_name=%s, shift_two_start=%s, shift_two_end=%s,
                    shift_three_name=%s, shift_three_start=%s, shift_three_end=%s
            """
            cursor.execute(update_query, (
                shift_one_name, shift_one_start, shift_one_end,
                shift_two_name, shift_two_start, shift_two_end,
                shift_three_name, shift_three_start, shift_three_end
            ))
        else:
            
            # Insert new row
            insert_query = """
                INSERT INTO shift_data (shift_one_name, shift_one_start, shift_one_end,
                                        shift_two_name, shift_two_start, shift_two_end,
                                        shift_three_name, shift_three_start, shift_three_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                shift_one_name, shift_one_start, shift_one_end,
                shift_two_name, shift_two_start, shift_two_end,
                shift_three_name, shift_three_start, shift_three_end
            ))

        db.commit()

        cursor.close()
        db.close()
        
        return jsonify({'message': 'Data saved successfully'}), 200

    except mysql.connector.Error as db_err:
        print(f"Database Error: {db_err}")
        return jsonify({"error": "Database Error", "message": str(db_err)}), 500

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Server Error", "message": str(e)}), 500






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cameras_json", type=str, default="cameras.json")
    parser.add_argument("--model", type=str, default="yolov8n-face.pt") # yolov8m.pt
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi_file", type=str, default="ROI.json")
    parser.add_argument("--video", type=str, help="C:/Users/SAMSUNG/Deevia/foot_fall_itc_project/backend_footfall_sanjeev/nag1.avi") 
    arguments = parser.parse_args()
    print("Initializing PeopleCounter...")
    people_counter = PeopleCounter(arguments)
    print("PeopleCounter initialized.")
    
    try:
        print("Starting server...")
        server = pywsgi.WSGIServer(('127.0.0.1',4200), app, handler_class=WebSocketHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    
    
    