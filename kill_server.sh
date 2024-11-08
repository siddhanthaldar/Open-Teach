#!/bin/bash
killall robot_camera.py
sudo fuser -k 5000/tcp
sudo fuser -k 10006/tcp
sudo fuser -k 10007/tcp
sudo fuser -k 10008/tcp
sudo fuser -k 10009/tcp
sudo fuser -k 11006/tcp
sudo fuser -k 11007/tcp
sudo fuser -k 11008/tcp
sudo fuser -k 11009/tcp