#!/bin/bash

PYTHON=/home/ubuntu/habitat_audit/DeepView.Predict/venv/bin/python
DATASET=/home/ubuntu/habitat_audit/new_dataset

for operation in linear bmm lstm conv2d conv_transpose2d; do
    $PYTHON train.py $operation ${DATASET}/${operation};
done
