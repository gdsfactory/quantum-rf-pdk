.PHONY: install-doc-fonts docs build install dev test

# Makefile — compatibility shim for gdsfactory PDK CI tooling
#
# This project uses `just` as its primary task runner (see justfile)
# and forwards all targets to just.

.DEFAULT_GOAL := "--list"

JUST_CMD := uvx --from rust-just just

# Font download URLs (Outfit & Inter from Google Fonts, Code New Roman from codeface)
FONT_BASE := https://github.com/chrissimpkins/codeface/raw/master/fonts/code-new-roman

install-doc-fonts:
	@if [ "$$GITHUB_ACTIONS" = "true" ]; then \
		sudo apt-get update -y && sudo apt-get install -y unzip && \
		mkdir -p /tmp/qpdk-fonts && \
		curl -fsSL "https://fonts.google.com/download?family=Outfit" -o /tmp/qpdk-fonts/outfit.zip && \
		curl -fsSL "https://fonts.google.com/download?family=Inter" -o /tmp/qpdk-fonts/inter.zip && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Regular.otf" -o /tmp/qpdk-fonts/CNR-Regular.otf && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Bold.otf" -o /tmp/qpdk-fonts/CNR-Bold.otf && \
		curl -fsSL "$(FONT_BASE)/Code%20New%20Roman-Italic.otf" -o /tmp/qpdk-fonts/CNR-Italic.otf && \
		sudo mkdir -p /usr/local/share/fonts/qpdk && \
		unzip -qo /tmp/qpdk-fonts/outfit.zip -d /tmp/qpdk-fonts/outfit && \
		unzip -qo /tmp/qpdk-fonts/inter.zip -d /tmp/qpdk-fonts/inter && \
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

# Any target not explicitly defined above is forwarded to just.
%:
	@$(JUST_CMD) $@
