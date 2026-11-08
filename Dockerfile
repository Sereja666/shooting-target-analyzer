FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb \
    LIBGL_ALWAYS_SOFTWARE=1

# Qt и часть X11 уже внутри колеса opencv-python. Снаружи нужны GL, glib и базовые X11.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libgomp1 \
        libice6 \
        libsm6 \
        libx11-6 \
        libxcb1 \
        libxext6 \
        novnc \
        python3-tk \
        python3-websockify \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 6080

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["--demo", "--no-sound"]
