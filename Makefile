.PHONY: all build clean convert-notebooks copy-sample-notebooks docs docs-latex docs-pdf git-rm-merged help install test test-fail-fast test-force update-pre write-cells

# Based on https://gist.github.com/prwhite/8168133?permalink_comment_id=4718682#gistcomment-4718682
help: ##@ (Default) Print listing of key targets with their descriptions
	@printf "\n\033[1;34mUsage:\033[0m \033[1;33mmake <command>\033[0m\n"
	@grep -F -h "##@" $(MAKEFILE_LIST) | grep -F -v grep -F | sed -e 's/\\$$//' | awk 'BEGIN {FS = ":*[[:space:]]*##@[[:space:]]*"}; \
	{ \
		if($$2 == "") \
			pass; \
else if($$0 ~ /^#/) \
			printf "\n%s\n", $$2; \
else if($$1 == "") \
			printf "     %-20s%s\n", "", $$2; \
else { \
			split($$1, arr, /[ :]/); \
			printf "\n    \033[34m%-20s\033[0m %s", arr[1], $$2; \
		} \
	}'

install: ##@ Install the package and all development dependencies
	uv sync --all-extras

CLEAN_DIRS := dist build *.egg-info docs/_build docs/notebooks
clean: ##@ Clean up all build, test, coverage and Python artifacts
	rm -rf $(CLEAN_DIRS)

###########
# Testing #
###########

test: ##@ Run the full test suite in parallel using pytest
	uv run pytest -n auto

test-gds: ##@ Run GDS regressions tests (tests/test_pdk.py)
	uv run pytest -s tests/test_pdk.py

test-gds-force: ##@ Run GDS regressions tests (tests/test_pdk.py) and regenerate
	uv run pytest -s tests/test_pdk.py --force-regen

test-gds-fail-fast: ##@ Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
	uv run pytest -s tests/test_pdk.py -x

update-pre: ##@ Update pre-commit hooks to the latest revisions
	uvx pre-commit autoupdate -j $$(expr $$(nproc) / 2 + $$(expr $$(nproc) % 2))

run-pre: ##@ Run all pre-commit hooks on all files
	uvx pre-commit run --all-files

git-rm-merged: ##@ Delete all local branches that have already been merged
	git branch -D `git branch --merged | grep -v \* | xargs`

build: ##@ Build the Python package (install build tool and create dist)
	rm -rf dist
	uv build

#################
# Documentation #
#################

write-cells: ##@ Write cell outputs into documentation notebooks (used when building docs)
	uv run .github/write_cells.py

convert-notebooks: ##@ Convert jupytext scripts from notebooks/src to ipynb format in notebooks
	./.github/convert-notebooks.sh

copy-sample-notebooks: ##@ Copy all sample scripts to use as notebooks docs
	mkdir -p docs/notebooks
	cp notebooks/src/*.py docs/notebooks/

docs: write-cells copy-sample-notebooks ##@ Build the HTML documentation
	uv run jb build docs

docs-latex: write-cells copy-sample-notebooks ##@ Setup LaTeX for PDF documentation
	uv run jb build docs --builder latex

docs-pdf: docs-latex ##@ Build PDF documentation (requires a TeXLive installation)
	cd "docs/_build/latex" && latexmk -pdfxe -xelatex -interaction=nonstopmode -f -file-line-error
