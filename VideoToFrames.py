import cv2
import os

# Read the video file
#vidcap = cv2.VideoCapture('C:\\Users\\OMOLP091\\Documents\\OMOTECH\RESEARCH\\TRAFFIC-PROJECT-SAMARTH_GUPTA\\TESTING-AMEY-SIR\\TESTING-RESULTS\\SCENARIO-3\\Scenario-3-Results-2.mp4')
vidcap = cv2.VideoCapture("C:\\Users\\OMOLP091\\Documents\\OMOTECH\RESEARCH\\VANIA_GOEL_BHARATHNATYAM_USING_OPENCV\\Thatti Mettu Adavu ｜ Kalakshetra Style ｜ Neha Chemmanoor.mp4")

# Get the frames per second (fps) of the video
fps = vidcap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Calculate the number of frames to skip (5 seconds interval)
frames_to_skip = int(fps * 2)
print(f"Frames to skip: {frames_to_skip}")

# Create an output directory to save the frames
output_dir = 'C:\\Users\\OMOLP091\\Documents\\OMOTECH\RESEARCH\\VANIA_GOEL_BHARATHNATYAM_USING_OPENCV\\video_output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the frame count
frame_count = 0
saved_frame_count = 0

while True:
    # Set the position of the next frame to capture
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    
    # Read the next frame from the video
    success, image = vidcap.read()
    if not success:
        break

    # Resize the image to 640x480 (optimal size is 180x180, adjust if necessary)
    resized_image = cv2.resize(image, (640, 480))

    # Save the frame as a JPEG image in the output directory
    frame_filename = f'{output_dir}/frame_{saved_frame_count:04d}.jpg'
    cv2.imwrite(frame_filename, resized_image)

    # Increment the frame count by the number of frames to skip
    frame_count += frames_to_skip
    saved_frame_count += 1

print("Total Frames Saved = ", saved_frame_count)
# Release the video capture object
vidcap.release()
