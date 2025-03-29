import streamlit as st
import requests
import time

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Title with reduced font size
st.markdown("<h3 style='text-align: center; color: black;'>Control Room Dashboard</h3>", unsafe_allow_html=True)

# Sidebar for video uploads and notifications
with st.sidebar:
    st.header("Upload Videos")
    uploaded_files = st.file_uploader(
        "Choose up to 6 video files", type=["mp4", "avi"], accept_multiple_files=True
    )
    
    # Button to trigger upload
    if st.button("Upload Videos"):
        if uploaded_files:
            # Limit to 6 videos
            uploaded_files = uploaded_files[:6]
            # Upload videos to the backend
            filenames = []
            for uploaded_file in uploaded_files:
                files = {"file": uploaded_file}
                response = requests.post(f"{BACKEND_URL}/upload/", files=files)
                if response.status_code == 200:
                    filename = response.json()["filename"]
                    filenames.append(filename)
                else:
                    st.error(f"Failed to upload video `{uploaded_file.name}`.")
            # Store filenames and initialize processed_videos in session state
            st.session_state["filenames"] = filenames
            st.session_state["processed_videos"] = set()  # Track videos where collisions have been reported
            # Rerun the app to switch to the video feed view
            st.rerun()
        else:
            st.warning("Please select video files before clicking 'Upload Videos'.")

    # Section for collision notifications
    st.subheader("Collision Notifications")
    if "collision_notifications" not in st.session_state:
        st.session_state["collision_notifications"] = []

# Main area for video feeds
if "filenames" in st.session_state:
    filenames = st.session_state["filenames"]
    processed_videos = st.session_state.get("processed_videos", set())  # Videos with detected collisions
    num_videos = len(filenames)

    # Debugging: Print the filenames to verify all videos are uploaded
    st.write(f"Uploaded {num_videos} videos: {filenames}")

    # Create a grid layout (2 rows x 3 columns for 6 videos)
    rows = [st.columns(3) for _ in range((num_videos + 2) // 3)]  # Adjust rows dynamically

    # Display video feeds in the grid with larger frames
    for i, filename in enumerate(filenames):
        row_idx = i // 3  # Determine the row index
        col_idx = i % 3   # Determine the column index
        with rows[row_idx][col_idx]:
            st.subheader(f"Feed {i + 1}")
            video_feed_url = f"{BACKEND_URL}/video_feed?filename={filename}"
            st.markdown(
                f'<div style="position: relative;">'
                f'<img src="{video_feed_url}" style="width:100%; height:150px; border: 2px solid #ccc; border-radius: 10px;">'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Check for collision detection and video completion in the background
    max_checks = 120  # Check for 120 seconds (adjust as needed)
    checks = 0

    while checks < max_checks:
        for filename in filenames:
            # Skip videos that have already been processed
            if filename in processed_videos:
                continue
            try:
                # Check video status
                status_response = requests.get(f"{BACKEND_URL}/video_status/{filename}", timeout=5)
                status_data = status_response.json()
                if status_data["status"] == "completed":
                    processed_videos.add(filename)
                    print(f"Video `{filename}` marked as completed.")
                    continue  # Stop polling for this video
                
                # Check for collision detection
                collision_response = requests.get(f"{BACKEND_URL}/check_collision/{filename}", timeout=5)
                collision_data = collision_response.json()
                print(f"Polling `{filename}`: {collision_data}")  # Debug print
                if collision_data["collision_detected"]:
                    # Add collision notification to the sidebar
                    notification = f"⚠️ Collision detected in `{filename}`!"
                    st.session_state["collision_notifications"].append(notification)
                    print(f"Notification added for `{filename}`.")  # Debug print
                    processed_videos.add(filename)  # Mark video as processed
            except Exception as e:
                st.error(f"Error checking collision for `{filename}`: {e}")
        
        # Update the sidebar with collision notifications
        with st.sidebar:
            for notification in st.session_state["collision_notifications"]:
                st.markdown(notification)

        # Stop polling if all videos have been processed
        if len(processed_videos) == len(filenames):
            break

        time.sleep(1)  # Check every second
        checks += 1

    # Save updated state to session state
    st.session_state["processed_videos"] = processed_videos