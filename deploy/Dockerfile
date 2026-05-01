# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ติดตั้ง uv จาก image หลัก
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ก๊อปปี้ไฟล์จัดการ package เพื่อลงล่วงหน้า (ช่วยให้ build ครั้งต่อไปเร็วขึ้น)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# ก๊อปปี้โค้ดและส่วนประกอบทั้งหมด
COPY src ./src
COPY api ./api
COPY mcp_server ./mcp_server
COPY eval ./eval

# สั่งให้โหลดโมเดล Reranker มาเก็บไว้ในเครื่องเลย (ป้องกันการเสียเวลาโหลดตอนรันจริง)
RUN uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 8000

# คำสั่งรัน API server เมื่อเปิด container
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]