install:
	uv sync --extra docs --extra dev

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

test:
	uv run pytest -n $$(nproc)

test-gds:
	uv run pytest -s tests/test_pdk.py

test-gds-force:
	uv run pytest -s tests/test_pdk.py --force-regen

test-gds-fail-fast:
	uv run pytest -s tests/test_pdk.py -x

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

write-cells:
	uv run .github/write_cells.py

docs: write-cells
	uv run jb build docs

docs-pdf: write-cells
	uv run jb build docs --builder pdflatex

.PHONY: all clean install test test-force test-fail-fast update-pre git-rm-merged build jupytext notebooks docs docs-pdf write-cells
