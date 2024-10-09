import argparse
import datetime
import json
import numpy as np
import cv2
from ultralytics import YOLO
import mysql.connector
import os
import joblib
import matplotlib.pyplot as plt

def load_model(model):
    return YOLO(model)

def load_svm_model(svm_model_path):
    return joblib.load(svm_model_path)

def plot_roi(latest_frame, polygon, linepoints):
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(latest_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.line(latest_frame, linepoints[0], linepoints[1], color=(0, 255, 255), thickness=2)

def plot_bbox_withid(latest_frame, box, id):
    str_id = str(int(id.item()))
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(latest_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(latest_frame, str_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

def inside_or_outside(box, linepoints):
    box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box_xyxy
    centroid = [(x1 + x2) // 2, (y1 + y2) // 2]
    if (centroid[1] < linepoints[0][1]) or (centroid[1] < linepoints[1][1]):
        return "outside"
    return "inside"

def footfall_condition(box, polygon, crossed_ids):
    box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box_xyxy
    centroid = [(x1 + x2) // 2, (y1 + y2) // 2]

    id = box.id[0]
    str_id = str(int(id.item()))

    if centroid[1] < (polygon[0][1] + polygon[1][1]) // 2:  # If the centroid is above the line
        if str_id not in crossed_ids:
            crossed_ids[str_id] = True
        return False, crossed_ids

    elif centroid[1] > (polygon[0][1] + polygon[1][1]) // 2:  # If the centroid is below the line
        if str_id in crossed_ids:
            if crossed_ids[str_id]:
                del crossed_ids[str_id]
                return True, crossed_ids
        return False, crossed_ids

def save_data_in_database(latest_frame, box, id, date_time):
    print("Saving data to database...")
    
    # Database connection parameters
    db_config = {
        'user': 'root',
        'password': 'SoujuK@1799',
        'host': 'localhost',
        'database': 'footfallCounter'
    }

    # Establish database connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS footfall (
        id INT AUTO_INCREMENT PRIMARY KEY,
        cropped_image LONGBLOB,
        identifier VARCHAR(255),
        timestamp DATETIME
    )
    """
    cursor.execute(create_table_query)

    # Crop the image with padding
    box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box_xyxy
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(latest_frame.shape[1], x2 + padding)
    y2 = min(latest_frame.shape[0], y2 + padding)

    cropped_image = latest_frame[y1:y2, x1:x2]

    # Encode the image to store in database
    _, buffer = cv2.imencode('.jpg', cropped_image)
    cropped_image_binary = buffer.tobytes()

    # Insert data into the database
    insert_query = "INSERT INTO footfall (cropped_image, identifier, timestamp) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (cropped_image_binary, id, date_time))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()

def get_roi(roi_file):
    with open(roi_file, "r") as f:
        roi_file_content = f.read()
        polygon = json.loads(roi_file_content)
        point1 = polygon["polygon"]["point1"]
        point2 = polygon["polygon"]["point2"]
        point3 = polygon["polygon"]["point3"]
        point4 = polygon["polygon"]["point4"]
        polygon_points = [point1, point2, point3, point4]

        line_point1 = polygon["line"]["point1"]
        line_point2 = polygon["line"]["point2"]
        line_points = [line_point1, line_point2]

        return polygon_points, line_points

class PeopleCounter():
    def __init__(self, args) -> None:
        self.entry_count = 0
        self.visible_inside = 0
        self.visible_outside = 0
        self.crossed_ids = {}
        self.frame_shape = (1280, 720)
        self.error_image = np.zeros((self.frame_shape[0], self.frame_shape[1], 3), dtype="uint8")
        self.latest_frame = np.ones((self.frame_shape[0], self.frame_shape[1], 3), dtype="uint8")
        
        # Initialize video file instead of camera threads
        self.video_path = args.video
        self.video = None
        if self.video_path:
            self.video = cv2.VideoCapture(self.video_path)
        
        self.model = load_model(args.model)
        self.svm_model = load_svm_model(args.svm_model)
        self.polygon, self.linepoints = get_roi(args.roi_file)

        # Create directory for saving frames if it doesn't exist
        self.output_dir = "output_frames"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.frame_counter = 0

    def run(self):
        self.visible_inside = 0
        self.visible_outside = 0
        plt.ion()  # Turn on interactive mode for matplotlib
        while True:
            ret, latest_frame = self.video.read()
            if not ret:
                break  # End of video file

            detections = self.model.track(latest_frame, persist=True, classes=0, conf=0.7)

            for objects in detections:
                boxes = objects.boxes
                for box in boxes:
                    try:
                        id = box.id[0]
                    except TypeError:
                        continue

                    box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box_xyxy

                    side = inside_or_outside(box, self.linepoints)
                    if side == "inside":
                        self.visible_inside += 1

                        count_condition, self.crossed_ids = footfall_condition(box, self.polygon, self.crossed_ids)
                        if count_condition:
                            self.entry_count += 1
                            # Uncomment to save data in the database
                            # save_data_in_database(latest_frame, box, id.item(), datetime.datetime.now())

                    else:  # outside
                        self.visible_outside += 1

                    plot_bbox_withid(latest_frame, box, id)
            plot_roi(latest_frame, self.polygon, self.linepoints)

            # Save the frame as an image file
            frame_filename = os.path.join(self.output_dir, f"frame_{self.frame_counter:04d}.jpg")
            cv2.imwrite(frame_filename, latest_frame)
            self.frame_counter += 1

            # Display the frame using matplotlib
            plt.imshow(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axis
            plt.show()
            plt.pause(0.001)  # Pause to allow the frame to render

            # Optional: Print progress or status
            print(f"Processed frame: {self.frame_counter}")

        self.video.release()
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Ensure the last frame is shown
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--model", type=str, default="yolov8l-face.pt")  # yolov8m.pt
    parser.add_argument("--svm_model", type=str, required=True, help="Path to the SVM model")
    parser.add_argument("--roi_file", type=str, default="ROI.json")
    arguments = parser.parse_args()

    counter_obj = PeopleCounter(arguments)
    counter_obj.run()
