#!/bin/bash

# Install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Clone Detic repository with submodules
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt
