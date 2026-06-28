python pitnn_plecs_pi_tuner.py --plot


# Recommended to be conservative:  Kp = 0.0523   Ki = 1.847
python pitnn_plecs_server.py --kp 0.0523 --ki 1.847 --device cpu

# For improved result
python pitnn_plecs_server.py --kp 0.5 --ki 10 --device cpu

# Or
python .\pitnn_plecs_server.py --checkpoint ".\pitnn_dab_checkpoint.pt" --kp 0.5 --ki 10 --device cpu

# For more tuning
python pitnn_plecs_server_raw_power.py --kp 0.8 --ki 10 --device cpu

# use this when troubleshooting
python .\pitnn_plecs_server.py --checkpoint ".\pitnn_dab_checkpoint.pt" --kp 0 --ki 0 --device cpu


# Kill server processes for troubleshooting
netstat -ano | findstr :9876
taskkill /PID 21448 /F
netstat -ano | findstr :9876
