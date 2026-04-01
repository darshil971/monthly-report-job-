# ============================================================
# Stage 1: Builder - install dependencies
# ============================================================
FROM continuumio/miniconda3 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n app python=3.10 -y
SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]

# Install Poetry
RUN pip install poetry

WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry lock --no-interaction && \
    poetry install --no-interaction --no-ansi --no-root

# Install Playwright chromium
RUN playwright install chromium && \
    playwright install-deps chromium

# Download FastText language detection model (for theme clustering)
RUN wget -q -O /app/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev \
    curl \
    fontconfig \
    # Playwright system dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Install Noto fonts for Unicode symbol rendering in PDFs
RUN mkdir -p /usr/share/fonts/noto && \
    curl -sL -o /usr/share/fonts/noto/NotoSans.ttf "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf" && \
    curl -sL -o /usr/share/fonts/noto/NotoSansSymbols.ttf "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols/NotoSansSymbols-Regular.ttf" && \
    curl -sL -o /usr/share/fonts/noto/NotoSansSymbols2.ttf "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf" && \
    fc-cache -f

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app

# Copy Playwright browsers from builder
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Copy FastText model from builder
COPY --from=builder /app/lid.176.bin /app/lid.176.bin

# Activate conda environment
ENV PATH=/opt/conda/envs/app/bin:$PATH
ENV CONDA_DEFAULT_ENV=app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FASTTEXT_MODEL_PATH=/app/lid.176.bin
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/ecom-review-app-8eb6e1990945.json
ENV FIREBASE_ADMIN_CREDENTIALS_PATH=/app/ecom-review-app-8eb6e1990945.json

WORKDIR /app

# Create tmp directory
RUN mkdir -p tmp

# Copy application code
COPY . .

# Copy entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "src"]
