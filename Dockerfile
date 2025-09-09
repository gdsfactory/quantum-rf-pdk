# Note, to download from ghcr.io you may need to authenticate with docker login, e.g.
#     echo $(gh auth token) | docker login ghcr.io -u "$(gh api user | jq -r .login)" --password-stdin
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Create user for binder
ARG NB_USER=notebook-user
ARG NB_UID=1001
ENV USER=${NB_USER} \
    HOME=/home/${NB_USER} \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_CACHE_DIR=${HOME}/.cache/uv

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    mkdir -p ${HOME} && \
    chown -R ${USER}:${USER} ${HOME} && \
    chown -R ${USER}:${USER} /usr/local/

WORKDIR ${HOME}
USER ${USER}

# Install dependencies with cache mount
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --all-extras && \
    uv tool install jupyterlab

# Copy source code and install project
COPY --chown=${USER}:${USER} . ${HOME}
RUN --mount=type=cache,uid=${NB_UID},gid=${NB_UID},target=${HOME}/.cache/uv \
    uv sync --all-extras

# Set PATH to include virtual environment
ENV PATH="${HOME}/.venv/bin:$PATH"

# Expose Jupyter Lab port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
