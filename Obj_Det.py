import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm
import subprocess

#ipd.Video('Game1.mp4')
"""
cap = cv2.VideoCapture('Game1.mp4')
frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
print("frame number= " , frame_num , " height= " , height , " width= "  ,width , " fps= " , fps)
cap.release()
"""
"""
cap = cv2.VideoCapture('Game1.mp4')
ret, img = cap.read()
print('Returned', ret, 'and image of shape ', img.shape)
def display_cv2_img(img, figsize=(10,10)):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_)
    ax.axis("off")

display_cv2_img(img)
plt.show()
cap.release()
"""

"""
fig, axs = plt.subplots(8,14, figsize=(40,30))
axs = axs.flatten()

cap = cv2.VideoCapture('Game1.mp4')
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 10 == 0:
        axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[img_idx].set_title(f'Frame: {frame}')
        axs[img_idx].axis('off')
        img_idx += 1
plt.tight_layout()
plt.show()
cap.release()
"""
"""
import cv2
import pandas as pd

# Open the video file
cap = cv2.VideoCapture('Game1.mp4')

# Initialize lists to hold data
data = []

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Extract frame details
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Append data to the list
        # burada her frame için değer kaydediliyor ve oyuncu duvar vs burada tanımlanmalı.
        data.append({
            'frame_number': frame_idx,
            'height': height,
            'width': width,
            'fps': fps
        })
        
        frame_idx += 1

    # Release the video capture object
    cap.release()

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('video_data.csv', index=False)
print("CSV file created successfully!")





labels = pd.read_csv('video_data.csv', low_memory=False)

video_labels = (
    labels.query('frame_number == "Game1"').reset_index(drop=True).copy())
video_labels["video_frame"] = (video_labels["frame_number"]*11.9).round().astype("int")
video_labels["height"].value_counts()
"""

