#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Starting scheduler..."
curl -s "${BASE_URL}/engine/start" | jq

echo "Trades before:"
curl -s "${BASE_URL}/trades" | jq

echo "Waiting 30 seconds for force-trade mode to generate paper trades..."
sleep 30

echo "Trades after:"
curl -s "${BASE_URL}/trades" | jq
