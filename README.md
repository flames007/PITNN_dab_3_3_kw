# Relationship between the different files
pitnn_dab.py          ← training, physics simulation, DABPhysics class
    │
    ├── pitnn_deploy.py --mode closed_loop   uses DABPhysics as simulated plant
    │                                         requires pitnn_dab.py + checkpoint.pt
    │
    ├── pitnn_deploy.py --mode hardware      real hardware loop with ADC/PWM stubs

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

# 1. Default — demo mode, no arguments needed
python pitnn_deploy.py

# 2. Explicitly choose a mode
python pitnn_deploy.py --mode demo

python pitnn_deploy.py --mode export # (For onnx you need to pip install onnx)

python pitnn_deploy.py --mode closed_loop

python pitnn_deploy.py --mode hardware

# 3. With optional parameters
python pitnn_deploy.py --mode closed_loop --Vref 800 --Pmax 50000 --cycles 500

python pitnn_deploy.py --mode hardware --duration 30

python pitnn_deploy.py --checkpoint my_other_checkpoint.pt --mode demo

# 4. Get help
python pitnn_deploy.py --help

# Export for Usage
python pitnn_deploy.py --mode export

python pitnn_inspect_exports.py