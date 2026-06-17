.PHONY: install serve test lint format up down ollama-pull

install:
	pip install -e ".[dev]"

serve:
	fuko serve

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

up:
	docker compose up -d

down:
	docker compose down

ollama-pull:
	docker compose exec ollama ollama pull bge-m3
