# -*- coding: utf-8 -*-
"""
猫娘大乱斗 — 本地对战匹配服务器
端口: 3001
启动: uvicorn server:app --host 0.0.0.0 --port 3001 --reload
      或直接: python server.py
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("battle_arena_server")
SERVER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 占位数据
# ---------------------------------------------------------------------------

# TODO: [羁绊列表接入] 以下为占位内容，待羁绊数据结构和 API 确定后替换为真实数据
PLACEHOLDER_BONDS: list[str] = [
    "与主人愉快的第一天",
    "主人和我陪伴的100小时",
    "主人夸我的第一次",
    "和主人一起看的第一次日出",
    "主人生病时我在身旁的那个夜晚",
]

# TODO: [虚拟对手] 本地单人调试用，真实对战由匹配队列填充
DUMMY_OPPONENTS: list[dict] = [
    {"nekoName": "迷路的猫娘",  "ownerName": "不知道主人在哪"},
    {"nekoName": "傲娇大猫猫",  "ownerName": "才、才不是为了你"},
    {"nekoName": "困困小猫咪",  "ownerName": "打盹中的铲屎官"},
    {"nekoName": "社恐猫猫",    "ownerName": "躲在角落的主人"},
]

# ---------------------------------------------------------------------------
# 应用与 CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="猫娘大乱斗匹配服务器", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 内存存储
# ---------------------------------------------------------------------------

# 等待匹配的玩家: player_id -> PlayerEntry dict
waiting_room: dict[str, dict] = {}
# 已匹配结果: player_id -> opponent snapshot dict
matched: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic 模型
# ---------------------------------------------------------------------------


class JoinRequest(BaseModel):
    nekoName: str = "未知猫娘"
    ownerName: str = "未知主人"
    avatar: Optional[str] = None
    # TODO: [羁绊列表接入] bonds 目前接受任意字符串列表，待数据结构确定后添加校验
    bonds: list[str] = PLACEHOLDER_BONDS


# ---------------------------------------------------------------------------
# 匹配逻辑
# ---------------------------------------------------------------------------

# 记录上一次虚拟对手，确保连续两次不重复
# 使用 dict 包装避免 global 声明问题
_state: dict = {"last_dummy_name": None}


def try_match() -> None:
    """若等待室有 ≥2 人则立即配对。"""
    ids = list(waiting_room.keys())
    if len(ids) < 2:
        return
    id_a, id_b = ids[0], ids[1]
    player_a = waiting_room.pop(id_a)
    player_b = waiting_room.pop(id_b)
    matched[id_a] = dict(player_b)
    matched[id_b] = dict(player_a)


async def schedule_dummy_match(player_id: str) -> None:
    """单人调试：3 秒后若未匹配则随机分配一个虚拟对手（确保不连续重复）。"""
    await asyncio.sleep(3)
    if player_id in waiting_room and player_id not in matched:
        player = waiting_room.pop(player_id)
        
        last_name = _state["last_dummy_name"]
        # 确保连续两次不重复
        available = [d for d in DUMMY_OPPONENTS if d["nekoName"] != last_name]
        if not available:  # 如果全部都被排除（理论上不会发生），则回退到全部
            available = DUMMY_OPPONENTS
        
        dummy_base = random.choice(available)
        _state["last_dummy_name"] = dummy_base["nekoName"]
        
        dummy = {**dummy_base, "avatar": None, "bonds": PLACEHOLDER_BONDS}
        matched[player_id] = dummy
        print(f"[调试] {player['nekoName']} 匹配虚拟对手：{dummy['nekoName']}")


# ---------------------------------------------------------------------------
# 奇遇铸造已搬到 local_server/forge_server（端口 3002）。
#   - /arena/forge-facts       → /forge/facts
#   - /arena/forge-card-story  → /forge/card-story
#   + 新增 /forge/inventory CRUD（替代旧 localStorage 持久化）
# 本文件只保留对战匹配相关路由（/arena/join /arena/status /arena/leave /health）。
# ---------------------------------------------------------------------------



@app.post("/arena/join")
async def join_arena(body: JoinRequest):
    """玩家加入大乱斗，上传羁绊列表，返回 playerId 及对手信息（若已匹配）。"""
    player_id = str(uuid.uuid4())
    entry = {
        "nekoName":  body.nekoName,
        "ownerName": body.ownerName,
        "avatar":    body.avatar,
        "bonds":     body.bonds,   # TODO: 替换为真实羁绊列表
        "joinedAt":  time.time(),
    }
    waiting_room[player_id] = entry
    print(f"[加入] {entry['nekoName']} ({player_id})  等待人数: {len(waiting_room)}")

    try_match()

    opponent = matched.get(player_id)
    if opponent is None:
        # 未立即匹配，安排虚拟对手兜底
        asyncio.create_task(schedule_dummy_match(player_id))

    return JSONResponse({"playerId": player_id, "opponent": opponent})


@app.get("/arena/status/{player_id}")
async def arena_status(player_id: str):
    """轮询匹配结果。"""
    opponent = matched.get(player_id)
    return JSONResponse({"opponent": opponent})


@app.post("/arena/leave/{player_id}")
async def arena_leave(player_id: str):
    """玩家离开，清理房间数据。"""
    waiting_room.pop(player_id, None)
    matched.pop(player_id, None)
    return JSONResponse({"ok": True})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "battle_arena_server"}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=3001, reload=True)
