.PHONY: all install check-lfs test clean help

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile).
# This Makefile provides the `install` and `test` targets expected by
# the gdsfactory PDK CI workflows (https://github.com/doplaydo/pdk-ci-workflow),
# and forwards every other target to just.

.DEFAULT_GOAL := help

all: install ## Default target (alias for install)

install: ## Install the package and all development dependencies
	uv sync --all-extras

check-lfs: ## Check if Git LFS is available and pull LFS files
	@if ! command -v git-lfs >/dev/null 2>&1; then \
	echo ""; \
	echo "Error: Git LFS is not installed!"; \
	echo ""; \
	echo "This repository uses Git LFS to store test data files."; \
	echo "Please install Git LFS before running tests:"; \
	echo ""; \
	echo "  Ubuntu/Debian:  sudo apt-get install git-lfs"; \
	echo "  RHEL/Rocky:     sudo dnf install git-lfs"; \
	echo "  macOS:          brew install git-lfs"; \
	echo "  Windows:        Download from https://git-lfs.github.com/"; \
	echo ""; \
	echo "After installing, run: git lfs install && git lfs pull"; \
	echo ""; \
	exit 1; \
	fi
	@echo "Git LFS is available. Pulling LFS files…"
	@git lfs pull

test: check-lfs ## Run the full test suite (honors PYTEST_ADDOPTS)
	uv run --extra models --extra graphics --extra hfss --extra qutip --group dev pytest -n auto

clean: ## Clean up build artifacts
	@just clean

help: ## Show this help message

	@echo "Makefile targets (compatibility shim — prefer 'just' for development):"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Any other target is forwarded to 'just <target>'."
	@echo "Run 'just --list' for all available commands."

# Any target not explicitly defined above is forwarded to just.
%:
	@just $@
