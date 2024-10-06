import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from tkinter import font
from tkinter import ttk

# List of four members to recognize
members = ["tharun", "praveen", "bhaskar", "rahil"]

# Dictionary to map member names to specific IDs (1 to 4)
names = {i+1: members[i] for i in range(len(members))}

# Step 1: Capture Faces
def capture_faces(name, save_dir='faces'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ensure the name is one of the four members
    if name not in members:
        messagebox.showwarning("Warning", "Name not in the list of recognized members!")
        return

    # Assign the ID based on the member's position in the list
    id = members.index(name) + 1

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    sample_num = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_num += 1
            # Save the captured face with the corresponding ID and sample number
            cv2.imwrite(f"{save_dir}/{id}_{sample_num}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sample_num >= 20:  # Take 20 images per person
            break

    cam.release()
    cv2.destroyAllWindows()

# Step 2: Train the Recognizer
def train_faces(data_dir='faces'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        
        # Ensure the filename contains a valid ID
        try:
            id = int(filename.split("_")[0])  # Get the ID from the filename
        except ValueError:
            print(f"Filename {filename} is not in the expected format!")
            continue

        # Load the image and convert it to grayscale
        gray_img = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_img, 'uint8')

        # Detect faces in the image
        faces = detector.detectMultiScale(img_numpy)

        # Add each face to the list along with its ID
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    ids = np.array(ids, dtype=np.int32)

    # Train the recognizer on the faces and IDs
    recognizer.train(face_samples, ids)
    recognizer.write('trainer.yml')

# Step 3: Recognize Faces in Real-time
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100 and id in names:
                name = names[id]  # Get the name from the dictionary using the ID
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            cv2.putText(img, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Step 4: Tkinter GUI
def start_capture():
    name = name_entry.get()
    if name:
        capture_faces(name)
        messagebox.showinfo("Info", "Face captured successfully!")
    else:
        messagebox.showwarning("Warning", "Please enter a name.")

def start_training():
    train_faces()
    messagebox.showinfo("Info", "Training completed!")

def start_recognition():
    recognize_faces()

# Set up the GUI
root = tk.Tk()
root.title("Face Recognition System")

# Load and set the background image
bg_image_path = r"C:/Users/shaik/OneDrive/Desktop/DALL·E 2024-09-17 20.58.40 - A background image for a face recognition system, featuring a soft gradient of light blue and pink tones. The image should have abstract patterns with.webp"
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((2000, 1000), Image.LANCZOS)  # Resize to match window
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image
background_label = tk.Label(root, image=bg_image_tk)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a big heading at the top
heading_font = font.Font(family='Helvetica', size=24, weight='bold')
heading_label = tk.Label(root, text="FACE RECOGNITION SYSTEM", fg='black', font=heading_font)
heading_label.place(relx=0.5, rely=0.1, anchor='center')

# Add some normal text instructions below the heading
instructions = tk.Label(root, text="Please enter your name and choose an action below.", bg='light pink', fg='black', font=('Helvetica', 14))
instructions.place(relx=0.5, rely=0.2, anchor='center')

# Create labels and buttons in the center of the window
tk.Label(root, text="Enter your name:", bg='light blue', fg='black').place(relx=0.5, rely=0.35, anchor='center')
name_entry = tk.Entry(root)
name_entry.place(relx=0.5, rely=0.4, anchor='center')

# Customize style for buttons with rounded corners using ttk
style = ttk.Style()
style.configure("Rounded.TButton",
                relief="flat",
                padding=10,
                font=("Helvetica", 14),
                background="blue",  # Button color
                foreground="black",  # Text color
                borderwidth=1)
style.map("Rounded.TButton",
          background=[("active", "light pink")])  # On-click color

# Make buttons bigger by setting width and height, and add rounded corners with 'ttk.Button'
capture_button = ttk.Button(root, text="Capture Face", command=start_capture, style="Rounded.TButton")
capture_button.place(relx=0.5, rely=0.5, anchor='center')

train_button = ttk.Button(root, text="Train Faces", command=start_training, style="Rounded.TButton")
train_button.place(relx=0.5, rely=0.6, anchor='center')

recognize_button = ttk.Button(root, text="Recognize Faces", command=start_recognition, style="Rounded.TButton")
recognize_button.place(relx=0.5, rely=0.7, anchor='center')

# Set window size and display the window
root.geometry("800x600")
root.mainloop()
