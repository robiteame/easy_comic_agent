import json

from fastapi import WebSocket


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        # project_id -> set of websockets
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        self.active_connections[project_id].add(websocket)

    def disconnect(self, websocket: WebSocket, project_id: str):
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)

    async def send_to_project(self, project_id: str, message: dict):
        """向项目的所有连接发送消息"""
        if project_id not in self.active_connections:
            return
        disconnected = set()
        for ws in self.active_connections[project_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        # 清理断开的连接
        for ws in disconnected:
            self.active_connections[project_id].discard(ws)


class JobsConnectionManager:
    """全局任务中心 WebSocket 连接管理器。

    与项目连接分开维护：任务事件是跨项目的，前端在 store 里按 project_id 决定
    是否采纳，因此这里不做项目过滤，也不会把旧项目的事件写进当前项目视图。
    广播只发送 DTO（不含 run token），并且对单个连接的发送失败是隔离的。
    """

    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    def has_connections(self) -> bool:
        return bool(self.connections)

    @property
    def connection_count(self) -> int:
        return len(self.connections)

    async def broadcast(self, message: dict) -> None:
        if not self.connections:
            return
        try:
            payload = json.dumps(message, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return
        disconnected = set()
        for ws in list(self.connections):
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.connections.discard(ws)


# 全局连接管理器
ws_manager = ConnectionManager()
jobs_manager = JobsConnectionManager()
