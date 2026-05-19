.PHONY: install-doc-fonts build install dev test nbdocs docs-pdf docs docs-serve doc drc-sample drc

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

nbdocs:
	rm -rf docs/notebooks/*.md
	find notebooks -maxdepth 1 -mindepth 1 -name "*.ipynb" | sort | \
	xargs -P4 -I{} uv run --extra docs jupyter nbconvert \
	--execute --to markdown --embed-images {} --output-dir docs/notebooks
	uv run python docs/hooks.py docs/notebooks/*.md

docs-pdf: nbdocs
	uv run python .github/write_cells.py
	uv run python .github/write_models.py
	cp CHANGELOG.md docs/changelog.md
	uv run mkdocs build -f mkdocs-pdf.yml

docs: nbdocs
	uv run python .github/write_cells.py
	uv run python .github/write_models.py
	cp CHANGELOG.md docs/changelog.md
	uv run --extra docs zensical build

docs-serve: nbdocs
	uv run python .github/write_cells.py
	uv run python .github/write_models.py
	cp CHANGELOG.md docs/changelog.md
	uv run --extra docs zensical serve -a localhost:8080
