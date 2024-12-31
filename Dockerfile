FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir flask opencv-python-headless matplotlib pyyaml tqdm requests psutil

EXPOSE 5000

ENTRYPOINT ["python3", "dumpling.py"]
