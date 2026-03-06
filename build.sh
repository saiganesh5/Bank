#!/usr/bin/env bash
set -o errexit

apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0

pip install -r requirements.txt
