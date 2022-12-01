FROM rocm/dev-ubuntu-22.04

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.2.2



RUN apt-get update
#RUN apt-get install wget -y
#RUN wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/jammy/amdgpu-install_5.3.50300-1_all.deb
#RUN apt-get install ./amdgpu-install_5.3.50300-1_all.deb -y

#RUN usermod -a -G render $LOGNAME


RUN apt-get install python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6  -y  # dep for cv2


RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /code

#COPY ./poetry.lock /code
COPY ./pyproject.toml /code
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2

COPY . /code

EXPOSE 8088
CMD ["uvicorn", "main:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "8088"]
