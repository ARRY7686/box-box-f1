FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Strategy model: trained at build time from ml/laps_train.csv (before the app
# files, so editing the app doesn't retrain it)
COPY ml/ ml/
RUN python ml/train_production.py

# Copy source files
COPY fetch_data.py serve.py simulator.html build_tracks.py track_layouts.json precache.py ./

# Bake a few showcase races into the image so they load instantly
RUN python precache.py

# Persistent cache directory (mount as a Fly.io volume at /app/race_cache).
# Hugging Face runs the container as user 1000, so the app must be able to write here.
RUN mkdir -p race_cache && chmod -R a+rwX race_cache race_seed

EXPOSE 7860

CMD ["python", "serve.py"]
