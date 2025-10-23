#!/bin/bash

# Start ML Trading Dashboard

echo "Starting ML Trading Dashboard..."
echo ""

# Set PYTHONPATH to current directory
export PYTHONPATH="$PWD"

# Create logs directory
mkdir -p logs

# Initialize database if not exists
if [ ! -f "data/trading.db" ]; then
    echo "Initializing database..."
    python -c "from src.database.models import init_database; init_database()"
fi

# Start API server in background
echo "Starting FastAPI backend on http://localhost:8000"
python src/api/main.py > logs/api.log 2>&1 &
API_PID=$!
echo "API started with PID $API_PID"

# Wait for API to start
sleep 3

# Start Streamlit dashboard
echo "Starting Streamlit dashboard on http://localhost:8501"
echo ""
echo "Dashboard will open in your browser..."
echo "Press Ctrl+C to stop both servers"
echo ""

streamlit run src/frontend/dashboard.py

# Cleanup: kill API server when streamlit exits
echo "Stopping API server..."
kill $API_PID
echo "Dashboard stopped"
