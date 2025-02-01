import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os


video_capture = cv2.VideoCapture(0)

# Load the known faces and their encodings
image_of_shraddha = face_recognition.load_image_file("faces/shraddha.jpg")
shraddha_encoding = face_recognition.face_encodings(image_of_shraddha)[0]

image_of_alia = face_recognition.load_image_file("faces/Alia.jpg")
alia_encoding = face_recognition.face_encodings(image_of_alia)[0]

image_of_bill = face_recognition.load_image_file("faces/Bill.jpg")
bill_encoding = face_recognition.face_encodings(image_of_bill)[0]

image_of_justin = face_recognition.load_image_file("faces/justin.jpg")
justin_encoding = face_recognition.face_encodings(image_of_justin)[0]

image_of_Priya = face_recognition.load_image_file("faces/Priya.jpg")
Priya_encoding = face_recognition.face_encodings(image_of_Priya)[0]


known_face_encodings = [shraddha_encoding, alia_encoding, bill_encoding, justin_encoding, Priya_encoding]
known_face_names = ["Shraddha", "Alia", "Bill Gates", "Justin", "Priya"]

#  initial list of students
students = known_face_names.copy()

date = datetime.now().strftime("%d-%m-%Y")
time = datetime.now().strftime("%H:%M:%S")
os.makedirs("Attendance", exist_ok=True)
f = open(f"Attendance/{date}.csv", "w+", newline="")
writer = csv.writer(f)
writer.writerow(["Name", "Date", "Time"])

face_locations = []
face_encodings = []
frame_counter = 0

try:
    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture frame from webcam. Exiting...")
            break

        frame_counter += 1
        if frame_counter % 5 != 0:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR frame to RGB format
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                font = cv2.FONT_HERSHEY_DUPLEX
                bottom_left_corner_of_text = (10, frame.shape[0] - 10)
                font_scale = 1.5
                font_color = (0, 255, 0)
                thickness = 2
                line_type = 2
                cv2.putText(frame, name + " Present", bottom_left_corner_of_text, font, font_scale, font_color,
                            thickness, line_type)

                # Mark attendance if the student is present
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([name, date, current_time])
                    print(f"Attendance registered: {name} at {current_time}")

                # Display the name on the video frame
                top, right, bottom, left = [i * 4 for i in face_location]  # Scale back up
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                # If face is not recognized
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "Unknown", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Attendance System", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Cleaning up...")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
