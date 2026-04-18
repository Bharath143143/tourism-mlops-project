# ── Base image: Python 3.10 slim for minimal container size
FROM python:3.10-slim

# ── Set working directory inside container
WORKDIR /app

# ── Create non-root user FIRST (HF Spaces requirement)
RUN useradd -m -u 1000 user

# ── Switch to non-root user before installing packages
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ── Copy and install dependencies first (cache layer optimization)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# ── Copy application source code with correct ownership
COPY --chown=user . $HOME/app

# ── Expose port 7860 (Hugging Face Spaces REQUIRED default)
EXPOSE 7860

# ── Launch Streamlit on container start
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
