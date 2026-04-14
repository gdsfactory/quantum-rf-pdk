# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile).
# This Makefile provides the `install` and `test` targets expected by
# the gdsfactory PDK CI workflows (https://github.com/doplaydo/pdk-ci-workflow),
# and forwards every other target to just.

.DEFAULT_GOAL := help

install: ## Install the package and all development dependencies
	uv sync --all-extras

test: ## Run the full test suite (honours PYTEST_ADDOPTS)
	uv run --extra models --extra graphics --extra hfss --extra qutip --group dev pytest -n auto

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

.PHONY: install test help
