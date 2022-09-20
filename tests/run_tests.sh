#!/bin/bash
# FOAM test executor 


# MODULE TESTS

# Ionosphere test 
python -m unittest test_ionosphere.py

# UTILS TESTS

# Reader test 
python -m unittest test_reader.py
