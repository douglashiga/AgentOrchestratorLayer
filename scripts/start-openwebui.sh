#!/bin/bash

# Start OpenAI-compatible API and Open WebUI

set -e

echo "🚀 Starting Agent Orchestrator with Open WebUI..."
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "✅ Created .env. You can edit it if needed."
    echo ""
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "📦 Building and starting services..."
echo ""

# Start all services with build
docker compose up --build -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if agent-api is healthy
echo "🔍 Checking agent-api health..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8010/health > /dev/null 2>&1; then
        echo "✅ agent-api is healthy"
        break
    fi
    attempt=$((attempt+1))
    if [ $attempt -lt $max_attempts ]; then
        echo "  Waiting... (attempt $attempt/$max_attempts)"
        sleep 1
    fi
done

if [ $attempt -eq $max_attempts ]; then
    echo "⚠️  agent-api health check timeout. It might still be starting..."
fi

echo ""
echo "🔍 Checking Open WebUI..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Open WebUI is ready"
        break
    fi
    attempt=$((attempt+1))
    if [ $attempt -lt $max_attempts ]; then
        echo "  Waiting... (attempt $attempt/$max_attempts)"
        sleep 1
    fi
done

echo ""
echo "📊 Service Status:"
docker compose ps
echo ""

# Get the models available
echo "📋 Available Models:"
curl -s http://localhost:8010/v1/models | grep -o '"id":"[^"]*"' || echo "  (models endpoint not yet ready)"
echo ""

echo "✅ Everything is set up!"
echo ""
echo "🌐 Open WebUI: http://localhost:3000"
echo "📡 API Endpoint: http://localhost:8010/v1"
echo ""
echo "💡 Tip: Select 'agent-orchestrator' or 'agent-orchestrator-fastpath' model in Open WebUI"
echo ""
echo "📖 For detailed setup, see: SETUP_OPENWEBUI.md"
echo ""

# Optional: Open in browser (macOS only)
if command -v open &> /dev/null; then
    read -p "Open Open WebUI in browser? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open http://localhost:3000
    fi
fi
