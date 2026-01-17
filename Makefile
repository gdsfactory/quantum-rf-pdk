.PHONY: all build clean convert-notebooks copy-sample-notebooks docs docs-latex docs-pdf git-rm-merged help install run-pre setup-ipython-config test test-fail-fast test-force test-gds test-gds-fail-fast test-gds-force update-pre write-cells write-makefile-help write-models

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
	uv sync --all-extras --all-groups

rm-samples: ##@ Remove samples folder
	rm -rf qpdk/samples

CLEAN_DIRS := dist build *.egg-info docs/_build docs/notebooks
clean: ##@ Clean up all build, test, coverage and Python artifacts
	rm -rf $(CLEAN_DIRS)

###########
# Testing #
###########

PYTEST_COMMAND := uv run --group dev pytest
test: ##@ Run the full test suite in parallel using pytest
	$(PYTEST_COMMAND) -n auto

test-gds: ##@ Run GDS regressions tests (tests/test_pdk.py)
	$(PYTEST_COMMAND) -s tests/test_pdk.py

test-gds-force: ##@ Run GDS regressions tests (tests/test_pdk.py) and regenerate
	$(PYTEST_COMMAND) -s tests/test_pdk.py --force-regen

test-gds-fail-fast: ##@ Run GDS regressions tests (tests/test_pdk.py) and stop at first failure
	$(PYTEST_COMMAND) -s tests/test_pdk.py -x

update-pre: ##@ Update pre-commit hooks to the latest revisions
	uvx prek autoupdate -j $$(expr $$(nproc) / 2 + $$(expr $$(nproc) % 2))

run-pre: ##@ Run all pre-commit hooks on all files
	uvx prek run --all-files

build: ##@ Build the Python package (install build tool and create dist)
	rm -rf dist
	uv build

#################
# Documentation #
#################

write-cells: ##@ Write cell outputs into documentation notebooks (used when building docs)
	uv run --group docs .github/write_cells.py

write-models: ##@ Write model outputs into documentation notebooks (used when building docs)
	uv run --group docs .github/write_models.py

write-makefile-help: ##@ Write Makefile help output to documentation
	uv run --group docs .github/write_makefile_help.py

convert-notebooks: ##@ Convert jupytext scripts from notebooks/src to ipynb format in notebooks
	./.github/convert-notebooks.sh notebooks/src/*.py

copy-sample-notebooks: ##@ Copy all sample scripts to use as notebooks docs
	mkdir -p docs/notebooks
	cp notebooks/src/*.py docs/notebooks/

setup-ipython-config: ##@ Setup IPython configuration for documentation build
	mkdir -p ~/.ipython/profile_default
	cp docs/ipython_config.py ~/.ipython/profile_default/ipython_config.py
	mkdir -p ~/.config/matplotlib/stylelib/
	cp docs/qpdk.mplstyle ~/.config/matplotlib/stylelib/qpdk.mplstyle

docs: write-cells write-models write-makefile-help copy-sample-notebooks ##@ Build the HTML documentation
	uv run --group docs jb build docs

docs-latex: write-cells write-models write-makefile-help copy-sample-notebooks ##@ Setup LaTeX for PDF documentation
	uv run --group docs jb build docs --builder latex

docs-pdf: docs-latex ##@ Build PDF documentation (requires a TeXLive installation)
	cd "docs/_build/latex" && XINDYOPTS="-M sphinx.xdy" latexmk -pdfxe -xelatex -interaction=nonstopmode -f -file-line-error || (test -f qpdk.pdf && echo "PDF generated despite warnings" && exit 0)
