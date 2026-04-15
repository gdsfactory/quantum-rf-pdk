.PHONY: docs build dev install test

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile)
# and forwards all targets to just.

.DEFAULT_GOAL := "--list"

JUST_CMD := uvx --from rust-just just

docs:
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then sudo apt-get update -y && sudo apt-get install -y fonts-cmu; fi
	@$(JUST_CMD) docs

build:
	@$(JUST_CMD) build

install:
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then \
		$(JUST_CMD) install "--all-extras --no-extra gdsfactoryplus"; \
	else \
		$(JUST_CMD) install; \
	fi

dev: install
	@$(JUST_CMD) install-pre

test:
	@$(JUST_CMD) test

# Any target not explicitly defined above is forwarded to just.
%:
	@$(JUST_CMD) $@
