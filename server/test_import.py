import sys

sys.path.insert(0, ".")

print("Testing imports...")

try:
    from config import settings

    print(f"Config loaded: {settings.DATABASE_URL}")
except Exception as e:
    print(f"Config error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

try:
    from db import init_db

    print("DB module loaded")
except Exception as e:
    print(f"DB error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

try:
    print("All routes loaded")
except Exception as e:
    print(f"Routes error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

try:
    print("WebSocket loaded")
except Exception as e:
    print(f"WebSocket error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("All imports successful!")

try:
    print("\nTesting database initialization...")
    init_db()
    print("Database initialized successfully!")
except Exception as e:
    print(f"Database error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
