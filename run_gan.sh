#!/bin/bash

echo "clean stuff "
rm snapshots/*
rm ac/*
rm *.png
rm *.pyc
rm *.hdf5

echo "start from scratch "

echo "convert data to hdf5 "
python data.py

echo "train gan "
python gan.py

echo "done!"
