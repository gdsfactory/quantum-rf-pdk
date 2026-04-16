.PHONY: docs build install dev test

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile)
# and forwards all targets to just.

.DEFAULT_GOAL := "--list"

JUST_CMD := uvx --from rust-just just

docs:
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then sudo apt-get update -y && sudo apt-get install -y fonts-cmu; fi
	@$(JUST_CMD) docs
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then \
		TOKEN=$${GH_TOKEN:-$${GITHUB_TOKEN:-$$(git config --get http.https://github.com/.extraheader | cut -d ' ' -f 3 | base64 -d | cut -d ':' -f 2 2>/dev/null)}}; \
		if [ -n "$$TOKEN" ]; then \
			GH_TOKEN=$$TOKEN gh run download $$GITHUB_RUN_ID -n pdf-docs -D docs/_build/html || echo "PDF artifact not found, skipping…"; \
		else \
			echo "GH_TOKEN not set and could not be extracted from git, skipping PDF download…"; \
		fi; \
	fi

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
