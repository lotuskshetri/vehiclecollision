# Vehicle Collision detection and notification system

Usage:
Create a virtual environment and install requirements.txt. This system was created with Python 3.12.5. 
Download model from this link: https://www.kaggle.com/models/lotuskshetri/vehicle-collision-detection. Keep it in the same directory.
To receive notification from Twilio, create twilio acc and sandbox environment.
Upto 6 vidoes can be uploaded at once. The videos are forced to run at 1fps. Notifications are displayed in real time.

Run the system with "uvicorn main:app --reload" and "streamlit run stream2.py" in another terminal

The model was built with Faster R-CNN architecture. It was pre-trained for vehicle detection with 4 classes then fine-tuned with 5 classes, the additional class being collision. 


![image](https://github.com/user-attachments/assets/4d0d0aee-5869-4dd7-bca7-e316c0977a4a)

