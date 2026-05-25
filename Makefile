.PHONY: install lint format test run docker-build docker-up

install:
	pip install -e ".[dev]" || pip install -r requirements-dev.txt

lint:
	ruff check semcode tests
	mypy semcode

format:
	black semcode tests scripts
	ruff check --fix semcode tests

test:
	pytest

run:
	uvicorn semcode.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t semcode:latest .

docker-up:
	docker compose up --build
