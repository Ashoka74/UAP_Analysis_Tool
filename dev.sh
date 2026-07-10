#!/bin/bash
# Development script to launch both API backend and frontend

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting UAP Data Analysis Tool (dev mode)${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
  echo -e "${YELLOW}🛑 Shutting down...${NC}"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  echo -e "${GREEN}✓ Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Check if backend dependencies are installed
if [ ! -d ".venv" ]; then
  echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
  uv sync
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
  echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
  cd frontend
  npm install
  cd ..
fi

echo ""
echo -e "${GREEN}📦 Starting FastAPI backend...${NC}"
echo -e "   → http://localhost:8000"
echo -e "   → API docs: http://localhost:8000/docs"
echo ""
echo -e "${GREEN}📦 Starting frontend dev server...${NC}"
echo -e "   → http://localhost:5173"
echo ""

# Start the backend
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start the frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✓ Both services running${NC}"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait
