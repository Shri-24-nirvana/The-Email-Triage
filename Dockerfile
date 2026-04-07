FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY graders/ ./graders/
COPY openenv.yaml ./
COPY inference.py ./

EXPOSE 7860

CMD ["python", "-m", "src.environment"]
