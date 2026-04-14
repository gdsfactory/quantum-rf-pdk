.PHONY: docs build dev

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile)
# and forwards all targets to just.

.DEFAULT_GOAL := "--list"

JUST_CMD := uvx --from rust-just just

docs:
	@$(JUST_CMD) docs

build:
	@$(JUST_CMD) build

dev:
	@$(JUST_CMD) install

# Any target not explicitly defined above is forwarded to just.
%:
	@$(JUST_CMD) $@
