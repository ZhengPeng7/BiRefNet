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
cp ./birefnet/evaluation/*.py ../${repo}/evaluation
cp ./birefnet/models/*.py ../${repo}/models
cp ./birefnet/models/backbones/*.py ../${repo}/models/backbones
cp ./birefnet/models/modules/*.py ../${repo}/models/modules
cp ./birefnet/models/refinement/*.py ../${repo}/models/refinement
cp -r ./.git* ../${repo}
