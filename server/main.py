import asyncio
import logging
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))

from api.routes import (  # noqa: E402
    asset,
    audio_track,
    budget,
    character,
    chat,
    graph,
    jobs,
    project,
    quality_review,
    regeneration_queue,
    render,
    script,
    shot,
    subtitle,
)
from api.routes import settings as settings_routes
from api.websocket import jobs_manager, ws_manager  # noqa: E402
from config import settings as app_settings  # noqa: E402
from db import SessionLocal, engine, init_db  # noqa: E402
from services import job_center  # noqa: E402
from services.error_reporter import install_log_redaction  # noqa: E402
from services.local_auth import (  # noqa: E402
    configured_token,
    is_allowed_websocket_origin,
    is_public_path,
    is_token_valid,
    request_token,
)
from services.task_registry import recover_interrupted  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn 会在启动时注册自己的 handler，这里再装一次脱敏过滤器。
    _configure_logging()
    init_db()
    recovered_deletions = project.recover_staged_project_deletions()
    if recovered_deletions:
        print(f"已恢复 {recovered_deletions} 个上次中断的项目删除事务")
    interrupted = recover_interrupted()
    if interrupted:
        print(f"已标记 {interrupted} 个服务重启中断的后台任务")
    print("数据库初始化完成")
    try:
        from services.model_config_service import apply_model_config_to_settings, migrate_store

        if migrate_store():
            print("旧版模型配置已自动迁移为端点式新格式")
        apply_model_config_to_settings()
        print("模型与 API 自定义配置已加载")
    except Exception as exc:  # noqa: BLE001
        print(f"模型与 API 自定义配置加载失败（沿用默认配置）: {exc}")
    # 启动前 Provider 预检：把视频端点的协议/模型/能力如实打进日志，
    # 不打印任何密钥内容；配置缺失时任务入口会给出结构化报错。
    try:
        from services.provider_readiness import video_provider_preflight

        preflight = video_provider_preflight()
        print(
            "视频 Provider 预检: protocol={protocol} model={model} base_url={base_url_host} "
            "api_key_configured={api_key_configured} reference_image={reference_image} "
            "reference_mode={reference_mode} native_audio={native_audio} "
            "fixed_duration={fixed_duration}".format(**preflight)
        )
        if preflight["issues"]:
            print("视频 Provider 预检未通过: " + "；".join(preflight["issues"]))
    except Exception as exc:  # noqa: BLE001
        print(f"视频 Provider 预检失败: {exc}")
    # 质量审核能力预检：自动模式的真实质量门禁依赖 VLM（可选身份 embedding）。
    # 未配置时如实打进日志——自动模式将拒绝自动批准并把镜头转人工。
    try:
        from services.quality_review_service import quality_review_service

        capability = quality_review_service.capability_summary()
        storyboard = capability["storyboard"]
        identity = storyboard.get("identity_embedding") or {}
        print(
            "质量审核能力预检: vlm_supported={vlm} identity_embedding_configured={identity} "
            "threshold={threshold} policy={policy}".format(
                vlm=storyboard.get("supported"),
                identity=identity.get("supported", False),
                threshold=capability["gate"]["threshold"],
                policy=capability["gate"]["policy"],
            )
        )
        if not storyboard.get("supported"):
            print(f"质量审核 VLM 未配置: {storyboard.get('reason') or '原因未知'}（自动模式不会自动批准镜头）")
        if identity and not identity.get("supported"):
            print(f"身份 embedding 未配置: {identity.get('reason') or '原因未知'}（身份相似度按未检测处理）")
    except Exception as exc:  # noqa: BLE001
        print(f"质量审核能力预检失败: {exc}")
    yield
    print("服务关闭")


app = FastAPI(
    title="AI 漫剧 Agent",
    description="AI 漫剧生产 Agent 后端服务",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 桌面客户端的渲染进程来源：开发态为 Vite Dev Server，打包态为 file:// (Origin 为 null)。
    # 不能与 allow_credentials=True 搭配使用通配符 "*"（浏览器会拒绝），因此显式列出来源。
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """输入校验失败时返回稳定的 422，且不回显原始输入。

    FastAPI 默认会把出错字段的原始值放进响应体；当该值是 JSON 扩展字面量
    NaN/Infinity（Python 的 json 模块可解析）时，响应序列化会失败并把本该是
    422 的请求变成 500。这里只保留字段路径、错误类型与可读原因。
    """

    errors = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "type": str(error.get("type", "")),
            "msg": str(error.get("msg", ""))[:200],
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def local_auth_middleware(request: Request, call_next):
    expected = configured_token()
    if (
        not expected
        or request.method == "OPTIONS"
        or is_public_path(request.url.path)
        or is_token_valid(expected, request.headers.get("x-comic-agent-token") or request.query_params.get("token"))
    ):
        return await call_next(request)
    return JSONResponse(status_code=401, content={"detail": "invalid local auth token"})


app.include_router(project.router)
app.include_router(asset.router)
app.include_router(script.router)
app.include_router(settings_routes.router)
app.include_router(shot.router)
app.include_router(render.router)
app.include_router(chat.router)
app.include_router(graph.router)
app.include_router(character.router)
app.include_router(jobs.router)
app.include_router(regeneration_queue.router, prefix="/api")
app.include_router(regeneration_queue.router, prefix="/api/shot")
app.include_router(budget.router)
app.include_router(subtitle.router)
app.include_router(audio_track.router)
app.include_router(quality_review.router)

output_dir = app_settings.OUTPUT_DIR
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")


async def _authorize_websocket(websocket: WebSocket) -> bool:
    """本地 API token 与 Origin 白名单校验；两个 WebSocket 入口共用。"""

    expected = configured_token()
    provided = request_token(dict(websocket.headers), websocket.query_params.get("token"))
    if not is_allowed_websocket_origin(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="websocket origin not allowed")
        return False
    if not is_token_valid(expected, provided):
        await websocket.close(code=1008, reason="invalid local auth token")
        return False
    return True


def jobs_snapshot() -> dict:
    """任务中心初始快照：连接建立后先发一次，避免错过历史事件。

    只包含最近的一批任务与统计，且与 REST 使用同一个 DTO —— 不含 run token。
    """

    db = SessionLocal()
    try:
        listing = job_center.list_jobs(
            db,
            job_center.JobQuery(page=1, page_size=app_settings.JOB_EVENT_SNAPSHOT_LIMIT),
        )
    finally:
        db.close()
    return {
        "type": "job_snapshot",
        "jobs": listing["items"],
        "total": listing["total"],
        "active_count": listing["active_count"],
        "status_counts": listing["status_counts"],
        "page": listing["page"],
        "page_size": listing["page_size"],
        "generated_at": listing["generated_at"],
    }


# 必须注册在 /ws/{project_id} 之前：否则 "jobs" 会被当成项目 ID 匹配掉。
@app.websocket("/ws/jobs")
async def jobs_websocket_endpoint(websocket: WebSocket):
    """全局任务中心事件通道：先发当前快照，再推送增量事件。"""

    if not await _authorize_websocket(websocket):
        return
    await jobs_manager.connect(websocket)
    try:
        await websocket.send_json(jobs_snapshot())
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            elif data == "sync":
                # 断线重连后前端会再要一次快照：只依赖增量事件无法保证一致性。
                await websocket.send_json(jobs_snapshot())
    except WebSocketDisconnect:
        jobs_manager.disconnect(websocket)
    except Exception:  # noqa: BLE001 - 单个连接异常不能影响其它连接
        jobs_manager.disconnect(websocket)


@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    if not await _authorize_websocket(websocket):
        return
    await ws_manager.connect(websocket, project_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, project_id)


@app.get("/")
async def root():
    return {"name": "AI 漫剧 Agent", "version": "0.1.0", "status": "running"}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    # The desktop shell uses this marker to distinguish ComicAgent from an
    # unrelated process that happens to occupy the fixed local API port.
    return {"status": "ok", "service": "comic-agent", "version": app.version}


@app.api_route("/livez", methods=["GET", "HEAD"])
async def livez():
    return {"status": "ok", "service": "comic-agent", "version": app.version}


@app.get("/readyz")
async def readyz():
    """Verify dependencies needed to accept render work, not just process liveness."""

    try:
        await asyncio.to_thread(_check_database)
        await asyncio.to_thread(_check_ffmpeg)
    except RuntimeError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "service": "comic-agent", "version": app.version, "database": "ok", "ffmpeg": "ok"}


def _check_database() -> None:
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("database unavailable") from exc


def _check_ffmpeg() -> None:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg unavailable")
    try:
        subprocess.run(
            [executable, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("ffmpeg unavailable") from exc


def _configure_logging() -> None:
    """全局日志配置：后台任务失败的堆栈需要落到 stderr，同时脱敏密钥。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    install_log_redaction()


def _start_parent_watchdog() -> None:
    """Exit when the desktop shell that spawned us disappears.

    The Electron main process keeps our stdin pipe open for its whole
    lifetime. If it dies without a chance to kill us (crash, SIGKILL), the
    pipe hits EOF and this daemon thread shuts the server down so a stale
    backend can never keep holding the database or a loopback port.
    """

    import threading

    def _watch() -> None:
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:
            return
        try:
            # EOF (and a broken-pipe read error, which is how a dead writer
            # can surface on Windows) both mean the shell is gone.
            while stream.read(4096) != b"":
                pass
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_watch, name="parent-watchdog", daemon=True).start()


if __name__ == "__main__":
    import os

    import uvicorn

    if os.getenv("COMIC_AGENT_PARENT_WATCH", "").strip() == "1":
        _start_parent_watchdog()

    # The desktop client only needs a local API. Allow an explicit HOST for
    # deployments that intentionally expose the service, but keep the default
    # private to avoid exposing API keys and generated media on the LAN.
    host = os.getenv("HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} and not configured_token():
        raise SystemExit("HOST 绑定到非本机地址时必须设置 COMIC_AGENT_LOCAL_TOKEN")
    uvicorn.run("main:app", host=host, port=int(os.getenv("PORT", "8011")), reload=False)
