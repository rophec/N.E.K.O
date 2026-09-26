(function () {
  "use strict";

  function jsonFetch(url, options) {
    const init = Object.assign({ credentials: "same-origin" }, options || {});
    init.headers = Object.assign({ "Content-Type": "application/json" }, init.headers || {});
    return fetch(url, init).then(async (response) => {
      const text = await response.text();
      let body = {};
      if (text) {
        try {
          body = JSON.parse(text);
        } catch (error) {
          body = { ok: false, reason: "invalid_json", raw: text };
        }
      }
      if (!response.ok) {
        body.ok = false;
        body.status = response.status;
      }
      return body;
    }).catch((error) => {
      return { ok: false, reason: "network_error", error: String(error && error.message ? error.message : error) };
    });
  }

  function makeSessionId(gameType) {
    return `${gameType || "game"}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function create(options) {
    const state = Object.assign({
      gameType: "game",
      lanlanName: "",
      sessionId: "",
      apiBase: "",
      heartbeatTimer: 0,
      heartbeatIntervalMs: 15000,
    }, options || {});
    if (!state.sessionId) state.sessionId = makeSessionId(state.gameType);
    state.apiBase = String(state.apiBase || "").replace(/\/+$/, "");

    function route(path) {
      return `${state.apiBase}/api/game/${encodeURIComponent(state.gameType)}${path}`;
    }

    function routeBody(extra) {
      return JSON.stringify(Object.assign({
        lanlan_name: state.lanlanName,
        session_id: state.sessionId,
      }, extra || {}));
    }

    const api = {
      state,
      startRoute(extra) {
        return jsonFetch(route("/route/start"), { method: "POST", body: routeBody(extra) });
      },
      heartbeat(extra) {
        return jsonFetch(route("/route/heartbeat"), { method: "POST", body: routeBody(extra) });
      },
      startHeartbeat() {
        api.stopHeartbeat();
        state.heartbeatTimer = window.setInterval(() => {
          api.heartbeat().catch((error) => console.warn("[NekoMiniGameSDK] heartbeat failed", error));
        }, state.heartbeatIntervalMs);
        return state.heartbeatTimer;
      },
      stopHeartbeat() {
        if (state.heartbeatTimer) {
          window.clearInterval(state.heartbeatTimer);
          state.heartbeatTimer = 0;
        }
      },
      sendEvent(event) {
        return jsonFetch(route("/chat"), { method: "POST", body: routeBody({ event: event || {} }) });
      },
      speak(line) {
        return jsonFetch(route("/speak"), { method: "POST", body: routeBody({ line: line || "" }) });
      },
      mirrorAssistant(line) {
        return jsonFetch(route("/mirror-assistant"), { method: "POST", body: routeBody({ line: line || "" }) });
      },
      endRoute(reason) {
        api.stopHeartbeat();
        return jsonFetch(route("/route/end"), { method: "POST", body: routeBody({ reason: reason || "game_end" }) });
      },
      submitScore(score, extra) {
        return jsonFetch(route("/leaderboard"), {
          method: "POST",
          body: routeBody(Object.assign({ finalScore: score }, extra || {})),
        });
      },
      loadCharacter() {
        const params = new URLSearchParams();
        if (state.lanlanName) params.set("lanlan_name", state.lanlanName);
        return jsonFetch(route(`/character?${params.toString()}`));
      },
      loadAudio(configUrl) {
        return jsonFetch(configUrl || `${state.apiBase}/static/game/games/${encodeURIComponent(state.gameType)}/${encodeURIComponent(state.gameType)}-audio-config.js`);
      },
    };
    return api;
  }

  window.NekoMiniGameSDK = { create, makeSessionId };
})();
