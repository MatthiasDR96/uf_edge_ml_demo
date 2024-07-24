#!/bin/bash

# Export variables
export LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/uf_edge_ml_demo/data"

# Start Label Studio
label-studio -p 8080 -db /uf_edge_ml_demo/label_studio.sqlite3 > /dev/null 2>&1 &

# Copy the SQLite database file
python3 ./src/app.py