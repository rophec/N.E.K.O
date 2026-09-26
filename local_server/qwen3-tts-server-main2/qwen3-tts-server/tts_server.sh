#!/usr/bin/env bash
# Qwen3-TTS Server 一键启停脚本
# 用法: ./tts_server.sh {start|stop|restart|status|log}

set -euo pipefail

# ── 配置 ──
PROJECT_DIR="."
PYTHON_BIN="${PYTHON_BIN:-python3}"
PID_FILE="${PROJECT_DIR}/.tts_server.pid"
LOG_FILE="./tts_server.log"
HOST="0.0.0.0"
PORT=8091
WAIT_TIMEOUT=120

# ── 检查服务是否运行 ──
is_running() {
    if [ -f "${PID_FILE}" ]; then
        local pid
        pid=$(cat "${PID_FILE}" 2>/dev/null || true)
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
        rm -f "${PID_FILE}"
    fi
    return 1
}

# ── 等待服务就绪 ──
wait_for_ready() {
    local elapsed=0
    echo -n "  等待引擎初始化"
    while [ "${elapsed}" -lt "${WAIT_TIMEOUT}" ]; do
        if ! is_running; then
            echo ""
            echo "  ❌ 进程已退出，请检查日志: ${LOG_FILE}"
            tail -5 "${LOG_FILE}" 2>/dev/null
            return 1
        fi
        local health
        health=$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null || echo "")
        if [ -n "${health}" ]; then
            local ready
            ready=$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('engine_ready',False))" 2>/dev/null || echo "False")
            if [ "${ready}" = "True" ]; then
                echo ""
                echo "  ✅ 服务就绪 (耗时 ${elapsed}s)"
                echo "  端点: http://localhost:${PORT}/v1/audio/speech"
                return 0
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""
    echo "  ⚠️ 等待超时 (${WAIT_TIMEOUT}s)，请检查日志: ${LOG_FILE}"
    return 1
}

# ── 启动 ──
do_start() {
    if is_running; then
        echo "⚠️  服务已在运行 (PID: $(cat ${PID_FILE}))"
        return 0
    fi

    echo "🚀 启动 Qwen3-TTS Server..."

    cd "${PROJECT_DIR}"
    nohup "${PYTHON_BIN}" -m server.main > "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "  PID: ${pid}"
    echo "  日志: ${LOG_FILE}"
    echo "  端口: ${PORT}"

    wait_for_ready
}

# ── 停止 ──
do_stop() {
    if ! is_running; then
        echo "⚠️  服务未在运行"
        return 0
    fi

    local pid=$(cat "${PID_FILE}")
    echo "🛑 停止 Qwen3-TTS Server (PID: ${pid})..."

    kill "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null && [ "${waited}" -lt 15 ]; do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "${pid}" 2>/dev/null; then
        echo "  强制终止..."
        kill -9 "${pid}" 2>/dev/null || true
        sleep 1
    fi

    rm -f "${PID_FILE}"
    echo "  ✅ 已停止"
}

# ── 重启 ──
do_restart() {
    do_stop
    sleep 2
    do_start
}

# ── 状态 ──
do_status() {
    if is_running; then
        local pid=$(cat "${PID_FILE}")
        echo "🟢 服务运行中 (PID: ${pid})"
        local health
        health=$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null || echo '{}')
        echo "  健康状态: ${health}"
    else
        echo "🔴 服务未运行"
        if [ -f "${LOG_FILE}" ]; then
            echo "  最近日志:"
            tail -3 "${LOG_FILE}" 2>/dev/null
        fi
    fi
}

# ── 日志 ──
do_log() {
    if [ -f "${LOG_FILE}" ]; then
        tail -f "${LOG_FILE}"
    else
        echo "⚠️  日志文件不存在: ${LOG_FILE}"
    fi
}

# ── 主入口 ──
case "${1:-}" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    restart) do_restart ;;
    status)  do_status  ;;
    log)     do_log     ;;
    *)
        echo "Qwen3-TTS Server 一键启停脚本"
        echo ""
        echo "用法: $0 {start|stop|restart|status|log}"
        echo ""
        echo "  start   启动服务（自动等待引擎就绪）"
        echo "  stop    停止服务（优雅关闭 → 强制终止）"
        echo "  restart 重启服务"
        echo "  status  查看服务状态"
        echo "  log     实时查看日志 (tail -f)"
        echo ""
        echo "配置 (环境变量):"
        echo "  QWEN3_TTS_MODEL_DIR   模型目录 (默认: /data/models/Qwen3-TTS-12Hz-1.7B-Hutao-GGUF)"
        echo "  QWEN3_TTS_PORT        监听端口 (默认: 8091)"
        echo "  QWEN3_TTS_ONNX_PROVIDER  ONNX EP (默认: CUDA)"
        exit 1
        ;;
esac
