#!/usr/env/bin bash

#python generate_labels.py
#python generate_labels.py --force_tokenization
python generate_labels_no_query.py
python generate_labels_no_query.py --force_tokenization
