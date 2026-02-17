#!/usr/bin/env bash
set -euo pipefail

export RUN_MODE=replay
export MARKET_DATA_PROVIDER=replay
export MARKET_DATA_REPLAY_PATH=${MARKET_DATA_REPLAY_PATH:-data/replay}
export MARKET_DATA_REPLAY_SPEED=${MARKET_DATA_REPLAY_SPEED:-20}
export TELEGRAM_ENABLED=false

echo "Replay environment configured:"
echo "  RUN_MODE=$RUN_MODE"
echo "  MARKET_DATA_PROVIDER=$MARKET_DATA_PROVIDER"
echo "  MARKET_DATA_REPLAY_PATH=$MARKET_DATA_REPLAY_PATH"
echo "  MARKET_DATA_REPLAY_SPEED=$MARKET_DATA_REPLAY_SPEED"
echo "  TELEGRAM_ENABLED=$TELEGRAM_ENABLED"
echo

echo "Starting docker compose..."
docker compose up --build
