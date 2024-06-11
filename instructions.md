## Teleop
- Start camera servers.
- `python teleop.py`

# Data collection
- `python data_collect.py demo_num=1`

# Process data
- Update `PATH_TO_REPO` and `FOLDER_NAME` variables in `preprocess.py` and `convert_to_pkl.py`.
- Save as frames: `python preprocess.py`
- Then save as pkl: `python convert_to_pkl.py`