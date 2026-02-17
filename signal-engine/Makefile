PYTHON ?= python
DAYS ?= 45
SYMBOL ?= ETHUSDT
OUT ?= data/replay

.PHONY: replay-eth-5m replay-download

replay-eth-5m:
	$(PYTHON) tools/replay/download_bybit_klines.py --symbol ETHUSDT --interval 5 --days $(DAYS) --out $(OUT)

replay-download:
	$(PYTHON) tools/replay/download_bybit_klines.py --symbol $(SYMBOL) --interval $(INTERVAL) --days $(DAYS) --out $(OUT)
