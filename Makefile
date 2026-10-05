.PHONY: sync test start-local start-debug

sync:
	@uv sync --frozen --extra cpu

test:
	@uv run --frozen --extra cpu pytest -q

start-local:
	@echo "Starting server..."
	@uv run --frozen --extra cpu --no-dev python main.py

start-debug:
	@echo "Starting server in debug mode..."
	@uv run --frozen --extra cpu python -m debugpy --listen 127.0.0.1:5678 main.py
