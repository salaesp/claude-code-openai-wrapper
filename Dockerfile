# claude-agent-sdk spawns the Claude Code CLI (a Node binary), so the image needs
# both Python (wrapper) and Node (CLI). Subscription auth lives in ~/.claude — mount
# it at runtime, do NOT bake it into the image.
FROM python:3.12-slim

# Node.js for the Claude Code CLI
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# non-root; ~/.claude is mounted here at runtime
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app
ENV HOME=/home/app

# WRAPPER_API_KEY and CLAUDE_CODE_OAUTH_TOKEN must be supplied at runtime (-e / compose).
# Nothing sensitive is baked into the image.
ENV CLAUDE_MODEL=claude-opus-5

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
