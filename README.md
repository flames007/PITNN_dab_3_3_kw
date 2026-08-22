# If having issues extracting the frame from the video
# Run the following and replace "Video_Project" with the name of your video file
winget install ffmpeg

ffmpeg -i Video_Project.mp4 -c:v libx264 -c:a copy -preset fast Video_Project_h264.mp4

# Run
# Single run with a fixed seed
python pitnn_dab.py --video Video_Project.mp4 --seed 0

# Five independent runs
python pitnn_dab.py --video Video_Project.mp4 --runs 5 --seed 0

python pitnn_dab.py --video Video_Project.mp4 # If using oscilloscope video as training data alongside the synthetic dataset

python pitnn_dab.py # If using only the synthetic dataset