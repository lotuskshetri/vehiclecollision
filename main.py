from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import os
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from twilio_notifier import send_whatsapp_notification  # Import the Twilio notifier

# Initialize FastAPI
app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "fasterrcnn_finetuned_epoch10.pth"

# Define the model architecture
num_classes = 5  # Bus, Truck, Motorcycle, Car, Collision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load the model state dictionary
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()

# Define class labels and colors
class_labels = ["Bus", "Truck", "Car", "Collision", "Motorcycle"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

# Dictionaries to track collision counters and status
collision_counters = {}  # Tracks collision counters for each video (0 or 1)
video_status = {}  # Tracks whether a video is "processing" or "completed"

# Process video and yield frames with bounding boxes at 1 FPS
def process_video(input_path, filename):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform inference only once per second (1 FPS)
        if frame_count % int(fps) == 0:
            img_tensor = F.to_tensor(frame).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                predictions = model(img_tensor)
            pred_boxes = predictions[0]["boxes"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()
            # Draw bounding boxes on the current frame
            for i in range(len(pred_scores)):
                if pred_scores[i] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, pred_boxes[i])
                    label_index = pred_labels[i] - 1
                    label = class_labels[label_index]
                    color = colors[label_index]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if label == "Collision":
                        # Update collision counter only if it's currently 0
                        if collision_counters.get(filename, 0) == 0:
                            collision_counters[filename] = 1  # Set counter to 1 (collision detected)
                            print(f"Collision Detected in video `{filename}`!")
            # Encode frame as JPEG and yield it for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            if not _:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1
    cap.release()
    # Mark video as "completed" after processing
    video_status[filename] = "completed"

# Endpoint to upload video
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    # Initialize collision counter and video status for the uploaded video
    collision_counters[file.filename] = 0
    video_status[file.filename] = "processing"  # Mark video as "processing"
    return {"message": "Video uploaded successfully", "filename": file.filename}

# Endpoint to check video status
@app.get("/video_status/{filename}")
async def video_status_endpoint(filename: str):
    if filename not in video_status:
        return JSONResponse(content={"status": "not_found"}, status_code=404)
    return JSONResponse(content={"status": video_status[filename]})

# Stream processed video
@app.get("/video_feed")
async def video_feed(filename: str):
    input_path = f"{UPLOAD_FOLDER}/{filename}"  # Use the uploaded video's filename
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return StreamingResponse(
        process_video(input_path, filename),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Endpoint to check collision status
@app.get("/check_collision/{filename}")
async def check_collision(filename: str):
    if filename not in collision_counters:
        return JSONResponse(content={"collision_detected": False}, status_code=404)
    
    # Check if the collision counter is 1 (collision detected but not yet reported)
    if collision_counters[filename] == 1:
        collision_counters[filename] = 2  # Mark as "already notified"
        print(f"Collision reported for `{filename}`.")
        
        # Send WhatsApp notification
        send_whatsapp_notification(filename)
        
        return JSONResponse(content={"collision_detected": True})
    
    # If the collision has already been reported, return False
    return JSONResponse(content={"collision_detected": False})