FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3-venv make latexmk texlive-xetex \
        texlive-fonts-recommended fonts-freefont-otf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

COPY . .

RUN sphinx-build -M html src _build/userguide -c userguide -W --keep-going -n \
    && sphinx-build -M latexpdf src _build/userguide -c userguide -W --keep-going -n \
    && cp _build/userguide/latex/stat_userguide.pdf _build/userguide/html/

CMD ["cp", "-R", "_build/userguide/html/.", "/output/"]
