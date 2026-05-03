"""Modal deployment — serverless FastAPI with Claude research agent."""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands("uv --version")
    .add_local_dir(".", remote_path="/app", ignore=[".venv", ".git", "__pycache__"], copy=True)
    .workdir("/app")
    # ใช้ uv pip install เพื่อลงแพ็กเกจทั้งหมดเข้า System Python โดยตรง
    .run_commands("uv pip install --system -e .")
    .run_commands(
        "python -c 'from sentence_transformers import CrossEncoder; CrossEncoder(\"BAAI/bge-reranker-base\")'"
    )
)

app = modal.App("deep-research-agent", image=image)


@app.function(
    secrets=[modal.Secret.from_name("deep-research-secrets")],
    timeout=300,
    memory=4096,
)
@modal.asgi_app()
def fastapi_app():
    from api.main import app as fastapi_app

    return fastapi_app
