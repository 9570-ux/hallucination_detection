FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install torch-geometric

RUN pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "clean_enhanced_app.py", "--server.address=0.0.0.0", "--server.port=8501"]