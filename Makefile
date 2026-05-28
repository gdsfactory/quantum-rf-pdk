.PHONY: install-doc-fonts docs build install dev test update-changelog

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile)
# and forwards all targets to just.

.DEFAULT_GOAL := "--list"

JUST_CMD := uvx --from rust-just just

# Font download URLs (Outfit from GitHub, Code New Roman from codeface; Inter via apt fonts-inter)
FONT_BASE := https://github.com/chrissimpkins/codeface/raw/master/fonts/code-new-roman
FIRA_MATH_URL := https://github.com/firamath/firamath/releases/download/v0.3.4/FiraMath-Regular.otf

install-doc-fonts:
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then \
		sudo apt-get update -y && sudo apt-get install -y fonts-inter && \
		mkdir -p /tmp/qpdk-fonts && \
		curl -fsSL "https://github.com/Outfitio/Outfit-Fonts/raw/main/fonts/ttf/Outfit-Regular.ttf" -o /tmp/qpdk-fonts/Outfit-Regular.ttf && \
		curl -fsSL "https://github.com/Outfitio/Outfit-Fonts/raw/main/fonts/ttf/Outfit-Bold.ttf" -o /tmp/qpdk-fonts/Outfit-Bold.ttf && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Regular.otf" -o /tmp/qpdk-fonts/CNR-Regular.otf && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Bold.otf" -o /tmp/qpdk-fonts/CNR-Bold.otf && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Italic.otf" -o /tmp/qpdk-fonts/CNR-Italic.otf && \
		curl -fsSL "$(FIRA_MATH_URL)" -o /tmp/qpdk-fonts/FiraMath-Regular.otf && \
		sudo mkdir -p /usr/local/share/fonts/qpdk && \
	find /tmp/qpdk-fonts -type f \( -name '*.ttf' -o -name '*.otf' \) -exec sudo cp {} /usr/local/share/fonts/qpdk/ \; && \
		sudo fc-cache -f && rm -rf /tmp/qpdk-fonts; \
	fi

docs: install-doc-fonts
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

update-changelog:
	claude -p "remove links and make a user friendly changelog from @CHANGELOG.md to @docs/changelog.md"

# Any target not explicitly defined above is forwarded to just.
%:
	@$(JUST_CMD) $@
