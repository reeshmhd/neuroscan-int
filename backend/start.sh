#!/bin/bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:$PORT
