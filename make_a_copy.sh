#!/bin/bash
# Set dst repo here.
repo=$1
mkdir ../${repo}
mkdir ../${repo}/evaluation
mkdir ../${repo}/models
mkdir ../${repo}/models/backbones
mkdir ../${repo}/models/modules
mkdir ../${repo}/models/refinement

cp ./*.sh ../${repo}
cp ./*.py ../${repo}
cp ./evaluation/*.py ../${repo}/evaluation
cp ./models/*.py ../${repo}/models
cp ./models/backbones/*.py ../${repo}/models/backbones
cp ./models/modules/*.py ../${repo}/models/modules
cp ./models/refinement/*.py ../${repo}/models/refinement
cp -r ./.git* ../${repo}
