DEV := $(shell grep -oP '^DEV=\K.*' .env)

install:
ifeq ($(DEV),True)
	poetry install
	pre-commit install
	mypy --install-types
else
	poetry install --no-dev
endif

format:
	black src/
	isort src/
	flake8 src/
	mypy src/
