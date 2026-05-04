install:
	uv sync --extra docs --extra dev

dev:
	uv sync --all-extras
	uv pip install -e .
	uv run pre-commit install

tech:
	python install_tech.py

update:
	pur

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

release:
	git push
	git push origin --tags

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	uv run python docs/write_cells.py
	uv run jb build docs

pics:
	uv run gfp build-pics

test:
	uv run gfp test

.PHONY: drc doc docs build
