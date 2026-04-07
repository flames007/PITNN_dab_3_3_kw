Files included:
- pitnn_dab_multivideo_refactor.py
- video_manifest_example.json

How to run:
1) Synthetic only:
   python pitnn_dab_multivideo_refactor.py

2) One video:
   python pitnn_dab_multivideo_refactor.py --video Video_Project.mp4

3) Many videos:
   python pitnn_dab_multivideo_refactor.py --video_manifest video_manifest_example.json

Main refactor changes:
- multi-video manifest support
- confidence scoring for extracted video samples
- group-aware split by video_id
- weighted video/synthetic sampling
- checkpoint provenance for manifest and sample counts
