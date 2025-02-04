from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from werkzeug.utils import secure_filename

# Set up Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "fasterrcnn_finetuned_epoch10.pth"
num_classes = 5  # Bus, Truck, Motorcycle, Car, Collision

# Initialize Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Define class labels
class_labels = ["1", "2", "Car", "Collision", "5"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


@app.route("/", methods=["GET", "POST"])
def index():
    collision_detected = False  # Flag to track collision detection
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Save the uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Process the video
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], "output_video.mp4")
        collision_detected = process_video(video_path, output_path)

        # Render the page with collision status
        return render_template("index.html", collision_detected=collision_detected)

    return render_template("index.html", collision_detected=False)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Original frame rate of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1, (frame_width, frame_height))  # Output video at 1 FPS
    collision_detected = False  # Flag to track collision detection

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only one frame per second (1 FPS)
        if frame_count % int(fps) == 0:  # Skip frames based on the original FPS
            # Convert frame to tensor
            img_tensor = F.to_tensor(frame).unsqueeze(0).to(DEVICE)

            # Perform inference
            with torch.no_grad():
                predictions = model(img_tensor)

            # Extract predictions
            pred_boxes = predictions[0]["boxes"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()

            # Draw bounding boxes and check for "Collision"
            for i in range(len(pred_scores)):
                if pred_scores[i] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    label_index = pred_labels[i] - 1
                    label = class_labels[label_index]
                    color = colors[label_index]

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {pred_scores[i]:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Check if "Collision" is detected
                    if label == "Collision":
                        collision_detected = True

            # Write frame to output video
            out.write(frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    return collision_detected


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)