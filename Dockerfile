FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.2

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update
RUN apt-get install python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6  -y  # dep for cv2

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /code

COPY ./poetry.lock /code
COPY ./pyproject.toml /code
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

COPY . /code

EXPOSE 8088
CMD ["uvicorn", "main:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8088"]