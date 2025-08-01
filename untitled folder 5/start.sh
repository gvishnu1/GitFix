#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Check if we're using Docker or local development
if [ "$1" = "docker" ]; then
    echo "Starting GitHub Code Monitor using Docker..."
    docker-compose up
else
    echo "Starting GitHub Code Monitor in development mode..."
    
    # Ensure Python virtual environment
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Run the FastAPI server in the background
    echo "Starting API server..."
    uvicorn main:app --reload --host ${HOST:-0.0.0.0} --port ${PORT:-8000} &
    API_PID=$!
    
    # Wait a moment for the API to start
    sleep 3
    
    # Run the Streamlit UI
    echo "Starting Streamlit UI..."
    streamlit run code_monitor/chat_interface/streamlit_app.py --server.port=8501
    
    # When Streamlit is closed, also terminate the API server
    kill $API_PID
fi