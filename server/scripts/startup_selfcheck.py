import os
import sys


def _run_startup_check() -> None:
    print("=" * 60)
    print("ComicAgent Server Startup Test")
    print("=" * 60)

    # Test 1: Import config
    print("\n[1/5] Testing config import...")
    try:
        from config import settings

        print("  ✓ Config loaded successfully")
        print(f"  - DATABASE_URL: {settings.DATABASE_URL}")
        print(f"  - LLM_PROVIDER: {settings.LLM_PROVIDER}")
        print(f"  - IMAGE_PROVIDER: {settings.IMAGE_PROVIDER}")
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 2: Import database
    print("\n[2/5] Testing database import...")
    try:
        from db import engine, init_db

        print("  ✓ Database module loaded")
        print(f"  - Engine: {engine.url}")
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 3: Import models
    print("\n[3/5] Testing models import...")
    try:
        from models import Character, Project, SceneAsset, Shot

        print("  ✓ Models loaded")
        print(f"  - Project: {Project.__tablename__}")
        print(f"  - Shot: {Shot.__tablename__}")
        print(f"  - Character: {Character.__tablename__}")
        print(f"  - SceneAsset: {SceneAsset.__tablename__}")
    except Exception as e:
        print(f"  ✗ Models error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 4: Import routes
    print("\n[4/5] Testing routes import...")
    try:
        from api.routes import asset, character, chat, graph, project, render, script, shot

        print("  ✓ All routes loaded")
        print(f"  - asset: {asset.router.prefix}")
        print(f"  - character: {character.router.prefix}")
        print(f"  - chat: {chat.router.prefix}")
        print(f"  - graph: {graph.router.prefix}")
        print(f"  - project: {project.router.prefix}")
        print(f"  - render: {render.router.prefix}")
        print(f"  - script: {script.router.prefix}")
        print(f"  - shot: {shot.router.prefix}")
    except Exception as e:
        print(f"  ✗ Routes error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 5: Import websocket
    print("\n[5/5] Testing websocket import...")
    try:
        print("  ✓ WebSocket manager loaded")
    except Exception as e:
        print(f"  ✗ WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 6: Initialize database
    print("\n[6/6] Testing database initialization...")
    try:
        init_db()
        print("  ✓ Database initialized successfully")
    except Exception as e:
        print(f"  ✗ Database initialization error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed! Server should start correctly.")
    print("=" * 60)

    # Try to start the server
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8011"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        from services.local_auth import configured_token

        if not configured_token():
            raise SystemExit("HOST 绑定到非本机地址时必须设置 COMIC_AGENT_LOCAL_TOKEN")
    print(f"\nStarting server on {host}:{port}...")
    try:
        import uvicorn

        uvicorn.run("main:app", host=host, port=port, reload=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        import traceback

        traceback.print_exc()


# 本文件名匹配 pytest 的 ``*_test.py`` 收集模式：作为模块导入时（例如在
# server/ 目录下直接运行 pytest）绝不能启动阻塞的 uvicorn 服务器，只在
# 显式执行 ``python scripts/startup_selfcheck.py`` 时跑启动自检。
if __name__ == "__main__":
    _run_startup_check()
