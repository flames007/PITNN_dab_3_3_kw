# 1. Default — demo mode, no arguments needed
python pitnn_deploy.py

# 2. Explicitly choose a mode
python pitnn_deploy.py --mode demo
python pitnn_deploy.py --mode export
python pitnn_deploy.py --mode closed_loop
python pitnn_deploy.py --mode hardware

# 3. With optional parameters
python pitnn_deploy.py --mode closed_loop --Vref 800 --Pmax 50000 --cycles 500
python pitnn_deploy.py --mode hardware --duration 30
python pitnn_deploy.py --checkpoint my_other_checkpoint.pt --mode demo

# 4. Get help
python pitnn_deploy.py --help