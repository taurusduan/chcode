"""
会话管理 — thread_id, checkpointer DB, 历史会话列表/加载/删除
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

console = Console()


class SessionManager:
    def __init__(self, workplace_path: Path):
        self.workplace_path = workplace_path
        self.sessions_dir = workplace_path / ".chat" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.thread_id = self._new_thread_id()

    def _new_thread_id(self) -> str:
        return f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def new_session(self) -> None:
        self.thread_id = self._new_thread_id()

    def set_thread(self, thread_id: str) -> None:
        self.thread_id = thread_id

    async def list_sessions(self, checkpointer: AsyncSqliteSaver) -> list[str]:
        """从 checkpointer 获取所有历史 thread_id"""
        try:
            await checkpointer.setup()
            async with checkpointer.lock:
                rows = await checkpointer.conn.execute_fetchall(
                    "SELECT DISTINCT thread_id FROM checkpoints"
                )
            return [row[0] for row in rows if row[0]]
        except Exception:
            return []

    async def delete_session(
        self, thread_id: str, checkpointer: AsyncSqliteSaver
    ) -> bool:
        """删除指定会话的所有数据"""
        try:
            await checkpointer.adelete_thread(thread_id)
            return True
        except Exception as e:
            console.print(f"[red]删除会话失败: {e}[/red]")
            return False
