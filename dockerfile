# Використання офіційного образу Python як базового
FROM python:3.10-slim

# Встановлення необхідних системних залежностей
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

# Встановлення MongoDB клієнта
RUN pip install pymongo

# Встановлення залежностей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіювання файлів програми
COPY . /app

# Встановлення робочого каталогу
WORKDIR /app

# Налаштування VNC
ENV DISPLAY=:1
CMD ["sh", "-c", "Xvfb :1 -screen 0 1024x768x16 & x11vnc -display :1 -forever -nopw & python main.py"]
