# Vehicle Collision detection and notification system

Usage:
Create a virtual environment and install requirements.txt. This system was created with Python 3.12.5. 
Download model from this link: https://www.kaggle.com/models/lotuskshetri/vehicle-collision-detection. Keep it in the same directory.

Run the system with "uvicorn main:app --reload" and "streamlit run stream2.py" in another terminal

The model was built with Faster R-CNN architecture. It was pre-trained for vehicle detection with 4 classes then fine-tuned with 5 classes, the additional class being collision. 
