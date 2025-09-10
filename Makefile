COMPOSE_FILE := devops/docker-compose.yaml

install: setup build run

## --------------------------------------
## Environment Setup
## --------------------------------------

setup:
	@echo "--- Setting up development environment ---"
	@mkdir -p data data/db data/files data/redis
	@echo "Directories created."
	@if [ ! -f .env ]; then \
		echo "Creating .env file from .env.example..."; \
		cp .env.example .env; \
	else \
		echo ".env file already exists. Skipping creation."; \
	fi
	@echo "--- Setup complete ---"

## --------------------------------------
## Docker Commands
## --------------------------------------

build:
	@echo "--- Building Docker images ---"
	@docker-compose -f $(COMPOSE_FILE) build
	@echo "--- Build complete ---"

run:
	@echo "--- Starting Docker containers ---"
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "--- Services are up and running! ---"
