# Note, to download from ghcr.io you may need to authenticate with docker login, e.g.
#     echo $(gh auth token) | docker login ghcr.io -u "$(gh api user | jq -r .login)" --password-stdin
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Create user for binder
ARG NB_USER=notebook-user
ARG NB_UID=1001
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER}
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_CACHE_DIR=${HOME}/.cache/uv

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    mkdir -p ${HOME} && \
    chown -R ${USER}:${USER} ${HOME} && \
    chown -R ${USER}:${USER} /usr/local/

# Apt dependencies for gdsfactory & KLayout
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libexpat1 libexpat1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
USER ${USER}

# First install only dependencies with cache mount
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --all-extras && \
    uv pip install jupyterlab jupytext

# Copy source code, install project and convert jupytext scripts to notebooks
COPY --chown=${USER}:${USER} . ${HOME}
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    uv sync --all-extras && \
    uv run jupytext --to ipynb qpdk/samples/**/*.py

# Set PATH to include virtual environment
ENV PATH="${HOME}/.venv/bin:$PATH"

# Expose Jupyter Lab port
EXPOSE 8888

SHELL ["/bin/bash", "-c"]
CMD ["/usr/local/bin/uv", "run", "--with", "jupyter", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
