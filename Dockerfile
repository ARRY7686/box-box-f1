FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY fetch_data.py serve.py simulator.html build_tracks.py track_layouts.json precache.py ./

# Bake a few showcase races into the image so they load instantly
RUN python precache.py

# Persistent cache directory (mount as a Fly.io volume at /app/race_cache).
# Hugging Face runs the container as user 1000, so the app must be able to write here.
RUN mkdir -p race_cache && chmod -R a+rwX race_cache race_seed

EXPOSE 7860

CMD ["python", "serve.py"]
