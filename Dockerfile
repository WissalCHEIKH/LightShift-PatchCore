FROM python:3.11-slim

# opencv-python-headless a besoin de libglib ; libgl1 pour opencv-python (installé par anomalib)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements-app.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

# docker compose build --build-arg INSTALL_PATCHCORE=true  -> ajoute torch CPU + anomalib (lourd)
ARG INSTALL_PATCHCORE=false
RUN if [ "$INSTALL_PATCHCORE" = "true" ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY src ./src
COPY app ./app

ENV PYTHONPATH=/code \
    MODEL_BACKEND=mock \
    STATE_DIR=/code/app_state

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
