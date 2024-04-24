DEV := $(shell grep -oP '^DEV=\K.*' .env)

install:
ifeq ($(DEV),True)
	poetry install
else
	poetry install --no-dev
endif
	pre-commit install
	mypy --install-types

format:
	black src/
	isort src/
	flake8 src/
	mypy src/
