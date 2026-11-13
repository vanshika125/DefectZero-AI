FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/defectzero-venv \
    PATH=/opt/defectzero-venv/bin:${PATH}

COPY .devcontainer/requirements-linux.lock /tmp/requirements-linux.lock

RUN python -m venv "$VIRTUAL_ENV"

# Install the tested CPU-only PyTorch wheels from the official CPU index.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "torch==2.15.0+cpu" \
    "torchvision==0.29.2+cpu"

# Use the complete tested Linux lock without re-resolving Torch dependencies.
RUN python -m pip install --no-cache-dir --no-deps \
    --index-url https://pypi.org/simple \
    --requirement /tmp/requirements-linux.lock \
    && python -m pip check \
    && python -c "import platform, sys, torch; assert sys.version_info[:2] == (3, 11), sys.version; assert platform.machine() == 'x86_64', platform.machine(); assert torch.cuda.is_available() is False; print('Python', sys.version.split()[0], '|', platform.machine(), '| torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

USER vscode
