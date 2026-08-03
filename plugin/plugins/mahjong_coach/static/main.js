const DEFAULT_PLUGIN_ID = 'mahjong_coach';

function resolvePluginId(locationLike = window.location) {
  // 优先使用宿主实际提供的 /plugin/<id>/ui/ 路径，兼容安装器生成的后缀 ID。
  // Prefer the host-provided /plugin/<id>/ui/ path so installer-suffixed IDs keep working.
  const pathMatch = String(locationLike?.pathname || '').match(/^\/plugin\/([^/]+)\/ui(?:\/|$)/);
  if (pathMatch?.[1]) {
    try {
      return decodeURIComponent(pathMatch[1]);
    } catch (_error) {
      return pathMatch[1];
    }
  }

  // 独立打开静态页面时允许显式传入 plugin_id；规范 ID 只作为最后兜底。
  // Allow an explicit plugin_id for standalone pages; use the canonical ID only as a final fallback.
  const queryPluginId = new URLSearchParams(String(locationLike?.search || '')).get('plugin_id');
  return String(queryPluginId || '').trim() || DEFAULT_PLUGIN_ID;
}

const PLUGIN_ID = resolvePluginId();
const RUNS_URL = '/runs';
const RUN_TIMEOUT_MS = 30000;

const statusLine = document.getElementById('statusLine');
const refreshBtn = document.getElementById('refreshBtn');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const startLiveBtn = document.getElementById('startLiveBtn');
const startYoloLiveBtn = document.getElementById('startYoloLiveBtn');
const stopLiveBtn = document.getElementById('stopLiveBtn');
const activeRecognitionMode = document.getElementById('activeRecognitionMode');
const legacyModeBtn = document.getElementById('legacyModeBtn');
const yoloModeBtn = document.getElementById('yoloModeBtn');
const imagePathInput = document.getElementById('imagePathInput');
const turnInput = document.getElementById('turnInput');
const buttonsInput = document.getElementById('buttonsInput');
const forceCheckpointInput = document.getElementById('forceCheckpointInput');
const keywordsInput = document.getElementById('keywordsInput');
const intervalInput = document.getElementById('intervalInput');
const overlayInput = document.getElementById('overlayInput');
const riverTrackingModeInput = document.getElementById('riverTrackingModeInput');
const tileRecognitionModeInput = document.getElementById('tileRecognitionModeInput');
const settlementEnabledInput = document.getElementById('settlementEnabledInput');
const settlementConfidenceInput = document.getElementById('settlementConfidenceInput');
const settlementFramesInput = document.getElementById('settlementFramesInput');
const settlementGapInput = document.getElementById('settlementGapInput');
const settlementConfigSummary = document.getElementById('settlementConfigSummary');
const roundWindInput = document.getElementById('roundWindInput');
const seatWindInput = document.getElementById('seatWindInput');
const doraTilesInput = document.getElementById('doraTilesInput');
const autoStartLiveInput = document.getElementById('autoStartLiveInput');
const windowCandidateSelect = document.getElementById('windowCandidateSelect');
const refreshWindowsBtn = document.getElementById('refreshWindowsBtn');
const saveCapturePrefsBtn = document.getElementById('saveCapturePrefsBtn');
const windowSelectionStatus = document.getElementById('windowSelectionStatus');
const rankInput = document.getElementById('rankInput');
const roomInput = document.getElementById('roomInput');
const riskToleranceInput = document.getElementById('riskToleranceInput');
const goalBiasInput = document.getElementById('goalBiasInput');
const callBiasInput = document.getElementById('callBiasInput');
const saveProfileBtn = document.getElementById('saveProfileBtn');
const playerNicknameInput = document.getElementById('playerNicknameInput');
const searchPlayerBtn = document.getElementById('searchPlayerBtn');
const playerCandidateSelect = document.getElementById('playerCandidateSelect');
const refreshPlayerBtn = document.getElementById('refreshPlayerBtn');
const confirmPlayerBtn = document.getElementById('confirmPlayerBtn');
const profileLookupStatus = document.getElementById('profileLookupStatus');
const profileSummary = document.getElementById('profileSummary');
const analysisRoundWindInput = document.getElementById('analysisRoundWindInput');
const analysisSeatWindInput = document.getElementById('analysisSeatWindInput');
const analysisDoraTilesInput = document.getElementById('analysisDoraTilesInput');
const analysisSource = document.getElementById('analysisSource');
const mainPlan = document.getElementById('mainPlan');
const planDetail = document.getElementById('planDetail');
const biasValue = document.getElementById('biasValue');
const defensePostureValue = document.getElementById('defensePostureValue');
const riskBudgetValue = document.getElementById('riskBudgetValue');
const tableContextPanel = document.getElementById('tableContextPanel');
const tableContextStatus = document.getElementById('tableContextStatus');
const tableScoreGrid = document.getElementById('tableScoreGrid');
const tableCounterSummary = document.getElementById('tableCounterSummary');
const tableStrategyImpact = document.getElementById('tableStrategyImpact');
const defenseCandidatePanel = document.getElementById('defenseCandidatePanel');
const defenseCandidateList = document.getElementById('defenseCandidateList');
const lastReason = document.getElementById('lastReason');
const confidenceValue = document.getElementById('confidenceValue');
const updateCount = document.getElementById('updateCount');
const targetList = document.getElementById('targetList');
const cautionList = document.getElementById('cautionList');
const handTiles = document.getElementById('handTiles');
const handCount = document.getElementById('handCount');
const meldTiles = document.getElementById('meldTiles');
const meldCount = document.getElementById('meldCount');
const opponentMeldTiles = document.getElementById('opponentMeldTiles');
const opponentMeldCount = document.getElementById('opponentMeldCount');
const riverTiles = document.getElementById('riverTiles');
const riverCount = document.getElementById('riverCount');
const roundArchiveSummary = document.getElementById('roundArchiveSummary');
const roundArchiveEmpty = document.getElementById('roundArchiveEmpty');
const roundArchiveContent = document.getElementById('roundArchiveContent');
const archiveRoundId = document.getElementById('archiveRoundId');
const archiveSettlement = document.getElementById('archiveSettlement');
const archiveTime = document.getElementById('archiveTime');
const archiveRiichi = document.getElementById('archiveRiichi');
const archivePlan = document.getElementById('archivePlan');
const archiveHandTiles = document.getElementById('archiveHandTiles');
const archiveHandCount = document.getElementById('archiveHandCount');
const archiveMeldTiles = document.getElementById('archiveMeldTiles');
const archiveMeldCount = document.getElementById('archiveMeldCount');
const archiveOpponentMeldTiles = document.getElementById('archiveOpponentMeldTiles');
const archiveOpponentMeldCount = document.getElementById('archiveOpponentMeldCount');
const archiveRiverTiles = document.getElementById('archiveRiverTiles');
const archiveRiverCount = document.getElementById('archiveRiverCount');
const roundArchiveJson = document.getElementById('roundArchiveJson');
const decisionType = document.getElementById('decisionType');
const decisionOutput = document.getElementById('decisionOutput');
const liveState = document.getElementById('liveState');
const liveFrame = document.getElementById('liveFrame');
const liveWindow = document.getElementById('liveWindow');
const liveError = document.getElementById('liveError');
const pipelineSummary = document.getElementById('pipelineSummary');
const pipelineSteps = document.getElementById('pipelineSteps');
const timingSummary = document.getElementById('timingSummary');
const timingLogBody = document.getElementById('timingLogBody');
const framePreview = document.getElementById('framePreview');
const framePreviewEmpty = document.getElementById('framePreviewEmpty');
const framePreviewState = document.getElementById('framePreviewState');
const framePreviewPath = document.getElementById('framePreviewPath');
const tableRegionPreview = document.getElementById('tableRegionPreview');
const tableRegionPreviewEmpty = document.getElementById('tableRegionPreviewEmpty');
const tableRegionPreviewState = document.getElementById('tableRegionPreviewState');
const settlementPreview = document.getElementById('settlementPreview');
const settlementPreviewEmpty = document.getElementById('settlementPreviewEmpty');
const settlementPreviewState = document.getElementById('settlementPreviewState');
let autoRefreshTimer = 0;
let previewLoading = false;
let lastPreviewPath = '';
let queuedPreviewPath = null;
let preferencesHydrated = false;

function setStatus(text) {
  statusLine.textContent = text || '';
}

function setRecognitionMode(mode) {
  const normalized = mode === 'yolo26' ? 'yolo26' : 'legacy';
  if (tileRecognitionModeInput) {
    tileRecognitionModeInput.value = normalized;
  }
  if (legacyModeBtn) {
    legacyModeBtn.classList.toggle('is-active', normalized === 'legacy');
    legacyModeBtn.setAttribute('aria-pressed', String(normalized === 'legacy'));
  }
  if (yoloModeBtn) {
    yoloModeBtn.classList.toggle('is-active', normalized === 'yolo26');
    yoloModeBtn.setAttribute('aria-pressed', String(normalized === 'yolo26'));
  }
  if (activeRecognitionMode) {
    activeRecognitionMode.textContent = normalized.toUpperCase();
    activeRecognitionMode.classList.toggle('is-yolo', normalized === 'yolo26');
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function compact(value, fallback = '-') {
  const text = String(value || '').trim();
  return text || fallback;
}

function firstNonEmptyList(...values) {
  // 选择第一组非空牌列表，避免空缓存遮住本帧识别结果。
  // Pick the first non-empty tile list so an empty cache does not hide frame perception.
  for (const value of values) {
    const list = Array.isArray(value) ? value.filter(Boolean) : [];
    if (list.length) {
      return list;
    }
  }
  return [];
}

function firstNonEmptyPiles(...values) {
  // 选择第一份真正有牌的牌河，避免 {} 作为真值盖住实时识别。
  // Pick the first river object with real tiles so {} does not mask live recognition.
  for (const value of values) {
    const piles = value && typeof value === 'object' ? value : {};
    const hasTiles = Object.values(piles).some((items) => Array.isArray(items) && items.length);
    if (hasTiles) {
      return piles;
    }
  }
  return {};
}

function cleanStrategyText(value) {
  let text = String(value || '').replace(/\s+/g, ' ').trim();
  [
    ['主线：', ''],
    ['保留：', ''],
    ['对子：', ''],
    ['路线选择：', ''],
    ['筒子占比很高', '筒子多'],
    ['万子占比很高', '万子多'],
    ['索子占比很高', '索子多'],
    ['保留同色块', '保留同色'],
    ['同色块', '同色'],
    ['做搭子', '找顺子'],
    ['先清', '先打'],
    ['不硬染', '别强做清一色'],
    ['吃碰杠', '鸣牌'],
    ['进听', '听牌'],
  ].forEach(([from, to]) => {
    text = text.replaceAll(from, to);
  });
  return text.trim();
}

function firstSentence(value) {
  const text = cleanStrategyText(value);
  const cutAt = ['，', '；', ';', '。'].map((mark) => text.indexOf(mark)).filter((index) => index >= 0);
  return cutAt.length ? text.slice(0, Math.min(...cutAt)).trim() : text;
}

function firstPrefixedValue(values, prefix) {
  const items = Array.isArray(values) ? values : [];
  const match = items.find((item) => String(item || '').startsWith(prefix));
  return match ? String(match).slice(prefix.length).trim() : '';
}

function listValues(value) {
  return Array.isArray(value) ? value.filter((item) => String(item || '').trim()) : [];
}

function extractAfter(value, keywords, stopMarkers) {
  const text = String(value || '');
  let start = -1;
  let keywordLength = 0;
  keywords.forEach((keyword) => {
    const index = text.indexOf(keyword);
    if (index >= 0 && (start < 0 || index < start)) {
      start = index;
      keywordLength = keyword.length;
    }
  });
  if (start < 0) {
    return '';
  }
  let tail = text.slice(start + keywordLength).replace(/^[\s：:]+/, '');
  const stopAt = stopMarkers.map((mark) => tail.indexOf(mark)).filter((index) => index >= 0);
  if (stopAt.length) {
    tail = tail.slice(0, Math.min(...stopAt));
  }
  return tail.trim();
}

function briefItems(value, limit = 4) {
  const text = cleanStrategyText(value).replace(/^[，、\s]+|[，、\s]+$/g, '');
  if (!text) {
    return '';
  }
  const items = text.split(/[、，,\s]+/).map((item) => item.trim()).filter(Boolean);
  if (items.length <= limit) {
    return items.length ? items.join('、') : text;
  }
  return `${items.slice(0, limit).join('、')}等${items.length}张`;
}

function strategyHeadline(plan, targets, fallback) {
  const targetItems = Array.isArray(targets) ? targets : [];
  const mainTarget = targetItems.find((item) => String(item || '').startsWith('主线：'));
  return firstSentence(mainTarget || plan) || fallback;
}

function strategyBrief(plan, detail, targets, cautions, fallback) {
  const targetItems = Array.isArray(targets) ? targets : [];
  const cautionItems = Array.isArray(cautions) ? cautions : [];
  const keep = briefItems(firstPrefixedValue(targetItems, '保留：') || extractAfter(plan, ['保留'], ['，先', '；', '。']), 4);
  const discard = briefItems(
    firstPrefixedValue(cautionItems, '优先清理：') || extractAfter(`${plan}。${detail}`, ['先打', '先清', '打：'], ['，', '；', '。']),
    4,
  );
  const lines = [];
  if (keep) {
    lines.push(`留：${keep}`);
  }
  if (discard) {
    lines.push(`打：${discard}`);
  }
  return lines.join('\n') || firstSentence(detail) || fallback;
}

function percent(value) {
  const number = Number(value || 0);
  return `${Math.round(Math.max(0, Math.min(1, number)) * 100)}%`;
}

function boundedNumber(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(minimum, Math.min(maximum, parsed));
}

function settlementRuntimeArgs() {
  const enabled = Boolean(settlementEnabledInput?.checked);
  const minConfidence = boundedNumber(settlementConfidenceInput?.value, 0.72, 0, 1);
  const confirmFrames = Math.round(boundedNumber(settlementFramesInput?.value, 2, 1, 8));
  const confirmMaxGapMs = Math.round(boundedNumber(settlementGapInput?.value, 2500, 200, 10000));
  return {
    settlement_recognition_enabled: enabled,
    settlement_min_confidence: minConfidence,
    settlement_confirm_frames: confirmFrames,
    settlement_confirm_max_gap_ms: confirmMaxGapMs,
  };
}

function renderSettlementConfigSummary() {
  if (!settlementConfigSummary) {
    return;
  }
  const settings = settlementRuntimeArgs();
  settlementConfigSummary.textContent = settings.settlement_recognition_enabled
    ? `已启用 · ${percent(settings.settlement_min_confidence)} · ${settings.settlement_confirm_frames} 帧 · ${settings.settlement_confirm_max_gap_ms} ms`
    : '已关闭';
}

async function fetchJson(url, init = {}, timeoutMs = RUN_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

async function callPlugin(entryId, args = {}, timeoutMs = RUN_TIMEOUT_MS) {
  const created = await fetchJson(RUNS_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plugin_id: PLUGIN_ID, entry_id: entryId, args }),
  }, timeoutMs);
  const runId = created.run_id || created.id;
  if (!runId) {
    throw new Error('run_id_missing');
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await sleep(250);
    const record = await fetchJson(`${RUNS_URL}/${runId}`, {}, Math.max(1000, deadline - Date.now()));
    if (record.status === 'succeeded') {
      const exported = await fetchJson(`${RUNS_URL}/${runId}/export`, {}, Math.max(1000, deadline - Date.now()));
      const item = (exported.items || []).find((candidate) => candidate.type === 'json' && candidate.json);
      const payload = item ? item.json : {};
      if (payload.success === false || payload.error) {
        throw new Error(payload.error?.message || payload.message || 'plugin_call_failed');
      }
      return payload.data || {};
    }
    if (['failed', 'canceled', 'timeout'].includes(record.status)) {
      throw new Error(record.error?.message || record.message || record.status);
    }
  }
  throw new Error('plugin_call_timeout');
}

function renderList(node, values, emptyText, className) {
  node.replaceChildren();
  const items = Array.isArray(values) ? values.filter((value) => String(value || '').trim()) : [];
  if (!items.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    node.appendChild(empty);
    return;
  }
  items.forEach((value) => {
    const item = document.createElement('span');
    item.className = className;
    item.textContent = String(value);
    node.appendChild(item);
  });
}

function renderTileRail(container, countNode, tiles, emptyText) {
  container.replaceChildren();
  const values = Array.isArray(tiles) ? tiles.filter(Boolean) : [];
  countNode.textContent = `${values.length} 张`;
  if (!values.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  values.forEach((tile) => {
    const node = document.createElement('span');
    node.className = `tile tile-${String(tile).slice(-1)}`;
    node.textContent = String(tile);
    container.appendChild(node);
  });
}

function renderHand(tiles) {
  renderTileRail(handTiles, handCount, tiles, '暂无手牌');
}

function renderMeldCollection(container, countNode, melds = [], emptyText = '暂无副露') {
  container.replaceChildren();
  const values = Array.isArray(melds) ? melds : [];
  countNode.textContent = `${values.length} 组`;
  if (!values.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  values.forEach((meld, index) => {
    const row = document.createElement('div');
    row.className = 'river-row';
    const label = document.createElement('span');
    label.className = 'river-player';
    label.textContent = `副露 ${index + 1}`;
    row.appendChild(label);
    const tiles = Array.isArray(meld?.tiles) ? meld.tiles : [];
    tiles.forEach((tileValue) => {
      const tile = String(tileValue || '').trim();
      if (!tile) {
        return;
      }
      const node = document.createElement('span');
      node.className = `tile tile-small tile-${tile.slice(-1)}`;
      node.textContent = tile;
      row.appendChild(node);
    });
    container.appendChild(row);
  });
}

function renderMelds(melds = []) {
  renderMeldCollection(meldTiles, meldCount, melds);
}

function renderOpponentMeldCollection(
  container,
  countNode,
  meldsByOwner = {},
  emptyText = '暂无对手副露',
) {
  container.replaceChildren();
  const seatOrder = ['left_opponent', 'top_opponent', 'right_opponent'];
  const seatLabels = {
    left_opponent: '上家',
    top_opponent: '对家',
    right_opponent: '下家',
  };
  const kindLabels = {
    chi: '吃',
    pon: '碰',
    kan: '杠',
    unknown: '待复核',
  };
  const owners = [
    ...seatOrder,
    ...Object.keys(meldsByOwner || {}).filter((owner) => !seatOrder.includes(owner)).sort(),
  ];
  const groups = owners.flatMap((owner) => (
    Array.isArray(meldsByOwner?.[owner])
      ? meldsByOwner[owner].map((meld) => ({ owner, meld }))
      : []
  ));
  countNode.textContent = `${groups.length} 组`;
  if (!groups.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }

  groups.forEach(({ owner, meld }, groupIndex) => {
    const row = document.createElement('div');
    row.className = 'river-row opponent-meld-row';
    row.dataset.owner = owner;
    const label = document.createElement('span');
    label.className = 'river-player opponent-meld-label';
    const kind = String(meld?.kind || 'unknown');
    const fallbackIndex = groupIndex + 1;
    label.textContent = `${seatLabels[owner] || owner} · ${kindLabels[kind] || kind} ${Number(meld?.meld_index || fallbackIndex)}`;
    row.appendChild(label);

    const observed = Array.isArray(meld?.observed_tiles) ? meld.observed_tiles : [];
    const corrections = Array.isArray(meld?.corrections) ? meld.corrections : [];
    const tiles = Array.isArray(meld?.tiles) ? meld.tiles : [];
    const calledTileIndex = meld?.called_tile_index === null || meld?.called_tile_index === undefined
      ? -1
      : Number(meld.called_tile_index);
    tiles.forEach((tileValue, tileIndex) => {
      const tile = String(tileValue || '').trim();
      if (!tile) {
        return;
      }
      const node = document.createElement('span');
      node.className = `tile tile-small tile-${tile.slice(-1)}`;
      if (calledTileIndex === tileIndex) {
        node.classList.add('tile-called');
      }
      const correction = corrections.find((item) => Number(item?.tile_index) === tileIndex);
      if (correction) {
        node.classList.add('tile-corrected');
        node.title = `原始识别 ${correction.from}，按横置叫牌与合法牌组修正为 ${correction.to}`;
      } else if (observed[tileIndex] && String(observed[tileIndex]) !== tile) {
        node.title = `原始识别 ${observed[tileIndex]}，修正为 ${tile}`;
      }
      node.textContent = tile;
      row.appendChild(node);
    });
    if (meld?.tile_identity_reliable === false) {
      row.classList.add('is-unreliable');
      row.title = '该组几何成立，但牌面组合尚未通过合法性复核';
    }
    container.appendChild(row);
  });
}

function renderOpponentMelds(meldsByOwner = {}) {
  renderOpponentMeldCollection(opponentMeldTiles, opponentMeldCount, meldsByOwner);
}

function renderRiverCollection(container, countNode, piles = {}, emptyText = '暂无牌河') {
  container.replaceChildren();
  const entries = Object.entries(piles || {});
  const total = entries.reduce((count, [, items]) => count + (Array.isArray(items) ? items.length : 0), 0);
  const claimed = entries.reduce(
    (count, [, items]) => count + (Array.isArray(items)
      ? items.filter((item) => Boolean(item?.claimed_into_meld)).length
      : 0),
    0,
  );
  countNode.textContent = claimed
    ? `${total} 张历史 · ${total - claimed} 张在河`
    : `${total} 张`;
  if (!total) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  entries.forEach(([player, items]) => {
    const row = document.createElement('div');
    row.className = 'river-row';
    const label = document.createElement('span');
    label.className = 'river-player';
    label.textContent = player;
    row.appendChild(label);
    (Array.isArray(items) ? items : []).forEach((item) => {
      const tile = String(item?.tile || '').trim();
      if (!tile) {
        return;
      }
      const node = document.createElement('span');
      node.className = `tile tile-small tile-${tile.slice(-1)}`;
      node.textContent = tile;
      if (item?.claimed_into_meld) {
        node.classList.add('tile-claimed');
        const caller = String(item?.claimed_by || '其他玩家');
        const kind = String(item?.claimed_meld_kind || '副露');
        node.title = `${tile} 已被 ${caller} 的 ${kind} 鸣走；仍保留为弃牌历史，不在剩余枚数中重复计数。`;
      }
      row.appendChild(node);
    });
    container.appendChild(row);
  });
}

function renderRiver(piles = {}) {
  renderRiverCollection(riverTiles, riverCount, piles);
}

function renderRoundArchive(data = {}) {
  const hasArchivePayload = Array.isArray(data.round_history)
    || Object.prototype.hasOwnProperty.call(data, 'last_round_archive');
  if (!hasArchivePayload) {
    return;
  }
  const history = Array.isArray(data.round_history) ? data.round_history : [];
  const archive = data.last_round_archive && typeof data.last_round_archive === 'object'
    ? data.last_round_archive
    : history.at(-1) || {};
  if (!archive.archive_id) {
    roundArchiveSummary.textContent = '暂无已结束小局';
    roundArchiveEmpty.hidden = false;
    roundArchiveContent.hidden = true;
    return;
  }

  const state = archive.state && typeof archive.state === 'object' ? archive.state : {};
  const settlement = archive.settlement && typeof archive.settlement === 'object' ? archive.settlement : {};
  const kindLabel = {
    win: '和牌',
    exhaustive_draw: '荒牌流局',
    abortive_draw: '途中流局',
    unknown: '未分类结算',
  }[settlement.kind] || '结算';
  const archivedAt = Number(archive.archived_at || 0);
  const timeLabel = archivedAt > 0 ? new Date(archivedAt * 1000).toLocaleString() : '-';
  const riichiPlayers = Array.isArray(state.riichi_players) ? state.riichi_players.filter(Boolean) : [];

  roundArchiveEmpty.hidden = true;
  roundArchiveContent.hidden = false;
  roundArchiveSummary.textContent = `${compact(archive.round_id, archive.archive_id)} · ${kindLabel} · 共 ${history.length || 1} 局`;
  archiveRoundId.textContent = `${compact(archive.round_id)} / ${archive.archive_id}`;
  archiveSettlement.textContent = `${kindLabel} · ${percent(settlement.confidence)}`;
  archiveTime.textContent = timeLabel;
  archiveRiichi.textContent = riichiPlayers.length ? riichiPlayers.join('、') : '无';
  archivePlan.textContent = compact(
    state.local_direction || state.local_plan || state.current_plan || state.opening_plan,
    '无已保存策略',
  );
  renderTileRail(archiveHandTiles, archiveHandCount, state.last_hand_tiles, '未保存手牌');
  renderMeldCollection(archiveMeldTiles, archiveMeldCount, state.last_melds, '无副露');
  renderOpponentMeldCollection(
    archiveOpponentMeldTiles,
    archiveOpponentMeldCount,
    state.last_opponent_melds,
    '无对手副露',
  );
  renderRiverCollection(archiveRiverTiles, archiveRiverCount, state.last_discard_piles, '无牌河数据');
  roundArchiveJson.textContent = JSON.stringify(archive, null, 2);
}

function statusText(status) {
  return {
    done: '完成',
    active: '进行中',
    waiting: '等待',
    skipped: '跳过',
    blocked: '异常',
  }[status] || '等待';
}

function acceptedDetectionCount(hand = {}) {
  return Array.isArray(hand.raw_detections)
    ? hand.raw_detections.filter((item) => item && item.accepted).length
    : 0;
}

function occupiedDetectionCount(hand = {}) {
  return Array.isArray(hand.raw_detections)
    ? hand.raw_detections.filter((item) => item && item.occupied).length
    : 0;
}

function pipelineStep(id, label, status, detail) {
  return { id, label, status, detail: compact(detail, '-') };
}

function buildPipelineSteps(data = {}) {
  const state = data.round_state || data.coach_state || data || {};
  const decision = data.last_decision || data || {};
  const perception = decision.perception || {};
  const hand = perception.hand || {};
  const meld = perception.meld || {};
  const action = perception.action || {};
  const river = perception.river || {};
  const settlement = perception.settlement || {};
  const opponentMeldsByOwner = firstNonEmptyPiles(
    state.last_opponent_melds,
    river.opponent_melds,
  );
  const opponentMeldCountValue = Object.values(opponentMeldsByOwner)
    .reduce((count, items) => count + (Array.isArray(items) ? items.length : 0), 0);
  const live = data.live || {};
  const handCountValue = Array.isArray(hand.hand_tiles)
    ? hand.hand_tiles.length
    : listValues(state.last_hand_tiles || decision.hand_tiles).length;
  const accepted = acceptedDetectionCount(hand);
  const occupied = occupiedDetectionCount(hand);
  const reason = String(hand.reason || decision.reason_codes?.[0] || '').trim();
  const running = Boolean(live.running);
  const hasFrame = Boolean(live.last_frame_path || decision.engine_meta?.source || decision.decision_type);

  let captureStatus = 'waiting';
  let captureDetail = '等待实战观察或手动分析';
  if (live.status === 'waiting_for_game') {
    captureStatus = 'active';
    captureDetail = '窗口已绑定，等待正式牌桌连续两帧确认';
  } else if (live.status === 'view_obstructed') {
    captureStatus = 'blocked';
    captureDetail = '菜单或其他窗口遮挡牌桌；当前对局数据已保留';
  } else if (['settlement_candidate', 'round_settlement', 'awaiting_next_round'].includes(live.status)) {
    captureStatus = 'done';
    captureDetail = `${live.frame_index || 0} 帧，正在处理小局边界`;
  } else if (live.status === 'verifying_new_round') {
    captureStatus = 'active';
    captureDetail = '遮挡已消失，正在复核是否进入新局';
  } else if (live.status === 'error') {
    captureStatus = 'blocked';
    captureDetail = live.last_error || '截图循环异常';
  } else if (live.status === 'waiting_for_window') {
    captureStatus = 'active';
    captureDetail = live.last_error || '正在寻找雀魂窗口';
  } else if (live.last_frame_path) {
    captureStatus = 'done';
    captureDetail = `${live.frame_index || 0} 帧，${live.last_capture_source || 'capture'}`;
  } else if (hasFrame) {
    captureStatus = 'done';
    captureDetail = decision.engine_meta?.source || '已有分析输入';
  } else if (running) {
    captureStatus = 'active';
    captureDetail = '等待第一帧截图';
  }

  const settlementPhase = String(settlement.phase || state.settlement_phase || live.status || 'playing');
  const settlementKind = {
    win: '和牌结算',
    exhaustive_draw: '荒牌流局',
    abortive_draw: '途中流局',
    unknown: '小局结算',
  }[settlement.kind || state.settlement_kind] || '结算';
  let settlementStatus = hasFrame ? 'done' : 'waiting';
  let settlementDetail = hasFrame ? '本帧未发现结算，继续跟踪本局' : '等待截图';
  if (settlementPhase === 'settlement_candidate') {
    settlementStatus = 'active';
    settlementDetail = `候选 ${Number(settlement.confirmation_frames || 0)}/${Number(settlement.required_frames || 2)} 帧，正在复核`;
  } else if (settlementPhase === 'settlement_latched' || live.status === 'round_settlement') {
    settlementStatus = 'active';
    settlementDetail = `${settlementKind}已确认；冻结上一局识别结果`;
  } else if (settlementPhase === 'awaiting_next_round' || live.status === 'awaiting_next_round') {
    settlementStatus = 'done';
    settlementDetail = `${settlementKind}已关闭；开始确认下一局手牌`;
  }

  let calibrationStatus = 'waiting';
  let calibrationDetail = '等待截图尺寸';
  if (hand.reason === 'missing_hand_tile_templates') {
    calibrationStatus = 'blocked';
    calibrationDetail = '截图分辨率暂未匹配校准';
  } else if (hand.reason === 'image_path_missing' || hand.reason === 'image_missing') {
    calibrationStatus = 'waiting';
    calibrationDetail = '还没有可读截图';
  } else if (hand.reason || hand.ok || handCountValue > 0) {
    calibrationStatus = 'done';
    calibrationDetail = '已套用精确或缩放校准';
  }

  let handStatus = 'waiting';
  let handDetail = '等待稳定手牌';
  if (hand.ok || handCountValue > 0 || String(hand.reason || '').startsWith('inferred_open_')) {
    handStatus = 'done';
    handDetail = `${handCountValue || accepted} 张，${hand.reason || 'stable_hand'}`;
  } else if (hand.reason === 'unstable_hand_count') {
    handStatus = accepted > 0 ? 'active' : 'waiting';
    handDetail = accepted > 0 ? `已识别 ${accepted} 张，继续确认` : `检测到 ${occupied} 个牌位`;
  } else if (hand.reason && !['fingerprint_match'].includes(hand.reason)) {
    handStatus = hand.reason.includes('missing') ? 'blocked' : 'waiting';
    handDetail = hand.reason;
  }

  let meldStatus = 'waiting';
  let meldDetail = '等待副露区域';
  if (meld.ok || Number(state.last_open_meld_count || 0) > 0 || String(hand.reason || '').startsWith('inferred_open_')) {
    meldStatus = 'done';
    meldDetail = `${Number(meld.open_meld_count || state.last_open_meld_count || 0)} 组副露`;
    if (String(hand.reason || '').startsWith('inferred_open_') && !meld.ok) {
      meldDetail += '，由手牌数量推断';
    }
  } else if (meld.reason === 'no_self_melds' || meld.reason === 'closed_hand_count_no_melds') {
    meldStatus = 'skipped';
    meldDetail = '未看到自己副露';
  } else if (meld.reason) {
    meldStatus = meld.reason.includes('unavailable') ? 'blocked' : 'waiting';
    meldDetail = meld.reason;
  }
  if (opponentMeldCountValue > 0) {
    const opponentDetail = `对手副露 ${opponentMeldCountValue} 组（已分座位）`;
    meldDetail = meldStatus === 'done' ? `${meldDetail}；${opponentDetail}` : opponentDetail;
    meldStatus = 'done';
  }

  let actionStatus = 'waiting';
  let actionDetail = '等待操作按钮';
  if (Array.isArray(decision.buttons) && decision.buttons.length) {
    actionStatus = 'done';
    actionDetail = decision.buttons.join(' / ');
  } else if (action.source === 'opening_hand_scan') {
    actionStatus = 'skipped';
    actionDetail = '开局先跳过按钮识别';
  } else if (action.source || Array.isArray(action.detected_buttons)) {
    actionStatus = 'done';
    actionDetail = action.detected_buttons?.length ? action.detected_buttons.join(' / ') : '未发现关键按钮';
  } else if (decision.decision_type === 'observe' && reason === 'fingerprint_match') {
    actionStatus = 'skipped';
    actionDetail = '画面未变化';
  }

  let riverStatus = 'waiting';
  let riverDetail = '等待牌河扫描';
  if (river.ok || Object.keys(river.discard_piles || state.last_discard_piles || {}).length) {
    riverStatus = 'done';
    riverDetail = river.reason || '已读取牌河/缓存';
  } else if (river.reason === 'opening_skips_river_scan' || river.reason === 'river_scan_not_due') {
    riverStatus = 'skipped';
    riverDetail = river.reason === 'opening_skips_river_scan' ? '开局不扫牌河' : '本帧未到检查点';
  } else if (river.reason) {
    riverStatus = river.reason.includes('disabled') ? 'skipped' : 'waiting';
    riverDetail = river.reason;
  }

  const hasPlan = Boolean(state.current_plan || state.opening_plan || state.local_plan || decision.suggestion);
  let strategyStatus = hasPlan ? 'done' : 'waiting';
  let strategyDetail = hasPlan ? compact(state.last_update_reason || decision.decision_type, '已生成建议') : '等待前置识别';
  if (decision.decision_type === 'observe' && !hasPlan && handStatus === 'blocked') {
    strategyStatus = 'blocked';
    strategyDetail = '前置识别未通过';
  } else if (decision.action_required) {
    strategyStatus = 'done';
    strategyDetail = `操作窗口：${decision.decision_type}`;
  }

  if (live.status === 'waiting_for_game') {
    calibrationStatus = 'skipped';
    calibrationDetail = '正式牌桌尚未确认';
    handStatus = 'skipped';
    handDetail = '场景门控中，不启动手牌识别';
    meldStatus = 'skipped';
    meldDetail = '场景门控中，不启动副露识别';
    actionStatus = 'skipped';
    actionDetail = '场景门控中，不采信按钮';
    riverStatus = 'skipped';
    riverDetail = '场景门控中，不启动牌河识别';
    strategyStatus = 'waiting';
    strategyDetail = '进入正式牌局后再生成策略';
  } else if (live.status === 'view_obstructed') {
    handStatus = 'blocked';
    handDetail = '当前画面不可可靠识别，暂停刷新';
    meldStatus = 'skipped';
    meldDetail = '遮挡中，保留上次结果';
    actionStatus = 'skipped';
    actionDetail = '遮挡中，不采信按钮';
    riverStatus = 'skipped';
    riverDetail = '遮挡中，牌河不清空也不追加';
    strategyStatus = hasPlan ? 'done' : 'waiting';
    strategyDetail = hasPlan ? '保留遮挡前的策略，暂不更新' : '等待牌桌重新可见';
  } else if (live.status === 'verifying_new_round') {
    handStatus = 'active';
    handDetail = '已发现显著不同的手牌，等待第二帧确认';
    strategyStatus = 'waiting';
    strategyDetail = '新局尚未确认，不覆盖上一局状态';
  } else if (live.status === 'settlement_candidate' || live.status === 'round_settlement') {
    calibrationStatus = 'skipped';
    calibrationDetail = '结算门控已接管本帧';
    handStatus = 'skipped';
    handDetail = '冻结上一局手牌';
    meldStatus = 'skipped';
    meldDetail = '冻结上一局副露';
    actionStatus = 'skipped';
    actionDetail = '结算中不采信操作按钮';
    riverStatus = 'skipped';
    riverDetail = '冻结上一局牌河，不追加也不清空';
    strategyStatus = hasPlan ? 'done' : 'skipped';
    strategyDetail = hasPlan ? '保留上一局策略作为结算记录' : '本帧不生成策略';
  } else if (live.status === 'awaiting_next_round') {
    handStatus = handCountValue >= 12 ? 'active' : 'waiting';
    handDetail = handCountValue >= 12
      ? `候选新手牌 ${handCountValue} 张，确认 ${Number(settlement.new_hand_confirmation_frames || 0)}/2 帧`
      : '等待 12 至 14 张稳定新手牌';
    actionStatus = 'skipped';
    actionDetail = '新局确认前不采信按钮';
    riverStatus = 'skipped';
    riverDetail = '新局确认前继续保留上一局牌河';
    strategyStatus = 'waiting';
    strategyDetail = '新手牌稳定两帧后重新生成';
  }

  return [
    pipelineStep('capture', '截图/窗口', captureStatus, captureDetail),
    pipelineStep('settlement', '结算边界', settlementStatus, settlementDetail),
    pipelineStep('calibration', '分辨率校准', calibrationStatus, calibrationDetail),
    pipelineStep('hand', '手牌识别', handStatus, handDetail),
    pipelineStep('meld', '副露识别', meldStatus, meldDetail),
    pipelineStep('action', '按钮识别', actionStatus, actionDetail),
    pipelineStep('river', '牌河/立直', riverStatus, riverDetail),
    pipelineStep('strategy', '策略输出', strategyStatus, strategyDetail),
  ];
}

function renderPipeline(data = {}) {
  const steps = buildPipelineSteps(data);
  pipelineSteps.replaceChildren();
  const blocked = steps.find((step) => step.status === 'blocked');
  const active = steps.find((step) => step.status === 'active');
  const waiting = steps.find((step) => step.status === 'waiting');
  const doneCount = steps.filter((step) => step.status === 'done' || step.status === 'skipped').length;
  const current = blocked || active || waiting || steps[steps.length - 1];
  pipelineSummary.textContent = blocked
    ? `卡在：${blocked.label}`
    : `${doneCount}/${steps.length}，当前：${current.label}`;
  steps.forEach((step, index) => {
    const item = document.createElement('div');
    item.className = `pipeline-step is-${step.status}`;
    const marker = document.createElement('span');
    marker.className = 'pipeline-marker';
    marker.textContent = String(index + 1);
    const body = document.createElement('div');
    body.className = 'pipeline-body';
    const title = document.createElement('div');
    title.className = 'pipeline-title';
    const name = document.createElement('strong');
    name.textContent = step.label;
    const badge = document.createElement('span');
    badge.textContent = statusText(step.status);
    title.append(name, badge);
    const detail = document.createElement('p');
    detail.textContent = step.detail;
    body.append(title, detail);
    item.append(marker, body);
    pipelineSteps.appendChild(item);
  });
}

function formatMs(value) {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return '-';
  }
  return `${number.toFixed(number >= 100 ? 0 : 1)}ms`;
}

function shortDecision(value) {
  return {
    opening_plan: '开局',
    coach_checkpoint: '检查点',
    call_window: '鸣牌',
    riichi_window: '立直',
    defense_alert: '防守',
    settlement_candidate: '结算候选',
    round_settlement: '结算确认',
    awaiting_next_round: '等新局',
    round_idle: '等下局',
    observe: '观察',
    error: '错误',
  }[value] || compact(value, '-');
}

// 渲染插件内后端耗时日志，避免必须另开命令行窗口。
// Render the in-plugin backend timing log so the user does not need a separate terminal.
function renderTimingLog(rows = []) {
  timingLogBody.replaceChildren();
  const items = Array.isArray(rows) ? rows.slice(-30).reverse() : [];
  if (!items.length) {
    timingSummary.textContent = '等待第一帧';
    const emptyRow = document.createElement('tr');
    const empty = document.createElement('td');
    empty.colSpan = 18;
    empty.className = 'timing-empty';
    empty.textContent = '实战启动后，这里会显示每一帧的定位、截图、识别、策略耗时。';
    emptyRow.appendChild(empty);
    timingLogBody.appendChild(emptyRow);
    return;
  }

  const latest = items[0];
  timingSummary.textContent = `最近 ${items.length} 条，最新整轮 ${formatMs(latest.loop_ms)}`;
  items.forEach((item) => {
    const row = document.createElement('tr');
    if (item.status === 'error') {
      row.className = 'is-error';
    } else if (item.status === 'waiting_for_window') {
      row.className = 'is-waiting';
    }
    [
      item.frame,
      compact(item.status),
      shortDecision(item.decision),
      compact(item.source),
      compact(item.tile_mode),
      compact(item.fallback_reason),
      compact(item.river_mode),
      compact(item.river_reason),
      formatMs(item.locate_ms),
      formatMs(item.capture_ms),
      formatMs(item.settlement_ms),
      formatMs(item.analyze_ms),
      formatMs(item.hand_ms),
      formatMs(item.meld_ms),
      formatMs(item.action_ms),
      formatMs(item.river_ms),
      formatMs(item.strategy_ms),
      formatMs(item.loop_ms),
    ].forEach((value, index) => {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      if (index >= 8) {
        cell.className = 'timing-number';
      }
      row.appendChild(cell);
    });
    timingLogBody.appendChild(row);
  });
}

function setSelectValue(input, value) {
  if (!input || value === undefined || value === null) return;
  const normalized = String(value);
  if ([...input.options].some((option) => option.value === normalized)) {
    input.value = normalized;
  }
}

function renderPreferences(data = {}, force = false) {
  const preferences = data.preferences || {};
  const profile = preferences.profile || data.config?.player_profile || {};
  const target = preferences.target || {};
  if (!preferencesHydrated || force) {
    autoStartLiveInput.checked = Boolean(preferences.auto_start_live);
    setSelectValue(rankInput, profile.rank || 'unknown');
    setSelectValue(roomInput, profile.room || 'unknown');
    setSelectValue(riskToleranceInput, profile.risk_tolerance || 'balanced');
    setSelectValue(goalBiasInput, profile.goal_bias || 'balanced');
    setSelectValue(callBiasInput, profile.call_bias || 'balanced');
    if (target.title) {
      const existing = [...windowCandidateSelect.options].find((item) => item.value === target.title);
      if (!existing) {
        windowCandidateSelect.add(new Option(`${target.title}（已保存）`, target.title));
      }
      windowCandidateSelect.value = target.title;
      windowSelectionStatus.textContent = `已保存：${target.title}`;
    }
    preferencesHydrated = true;
  }
  const sourceLabel = profile.source === 'amae_koromo' ? '牌谱屋建议' : '手动';
  const confirmedLabel = profile.confirmed ? '已确认' : '待确认';
  profileSummary.textContent = `${sourceLabel} · ${confirmedLabel} · ${profile.risk_tolerance || 'balanced'}`;
}

function renderDefenseCandidates(strategy = {}) {
  const candidates = Array.isArray(strategy.top_candidates)
    ? strategy.top_candidates.slice(0, 3)
    : Array.isArray(strategy.candidates) ? strategy.candidates.slice(0, 3) : [];
  defenseCandidateList.replaceChildren();
  defenseCandidatePanel.hidden = candidates.length === 0;
  candidates.forEach((item, index) => {
    const card = document.createElement('article');
    card.className = 'defense-candidate-card';
    const title = document.createElement('strong');
    title.textContent = `#${index + 1} 打 ${compact(item.tile, '?')}`;
    const safety = document.createElement('span');
    safety.textContent = `${compact(item.safety, '安全度未知')} · 危险 ${Number(item.defense_risk || 0).toFixed(1)}`;
    const shape = document.createElement('span');
    const effectiveDelta = Number(item.effective_count_delta || 0);
    const effectiveDeltaText = `${effectiveDelta >= 0 ? '+' : ''}${effectiveDelta}`;
    shape.textContent = `向听 ${Number(item.shanten ?? 8)} · 牌型损失 ${Number(item.shape_loss || 0)} · 有效牌 ${Number(item.effective_count || 0)} 张（${effectiveDeltaText}）`;
    card.append(title, safety, shape);
    defenseCandidateList.appendChild(card);
  });
}

function renderTableContext(state = {}, strategy = {}) {
  const context = strategy.table_context && typeof strategy.table_context === 'object'
    ? strategy.table_context
    : {};
  const scores = context.scores && typeof context.scores === 'object'
    ? context.scores
    : state.player_scores || {};
  const ranks = context.ranks && typeof context.ranks === 'object'
    ? context.ranks
    : state.player_ranks || {};
  const entries = [
    ['self', '自己'],
    ['left_opponent', '上家'],
    ['top_opponent', '对家'],
    ['right_opponent', '下家'],
  ].filter(([key]) => Number.isFinite(Number(scores[key])));
  tableContextPanel.hidden = entries.length !== 4;
  tableScoreGrid.replaceChildren();
  if (entries.length !== 4) {
    return;
  }
  entries.forEach(([key, label]) => {
    const score = Number(scores[key]);
    const rank = Number(ranks[key] || 0);
    const tied = Object.values(scores).filter((value) => Number(value) === score).length > 1;
    const card = document.createElement('article');
    card.className = `table-score-card${key === 'self' ? ' is-self' : ''}`;
    const seat = document.createElement('span');
    seat.textContent = label;
    const points = document.createElement('strong');
    points.textContent = score.toLocaleString('zh-CN');
    const placement = document.createElement('em');
    placement.textContent = rank ? `${tied ? '并列' : ''}${rank}位` : '顺位待定';
    card.append(seat, points, placement);
    tableScoreGrid.appendChild(card);
  });
  const honba = context.honba_count ?? state.honba_count;
  const sticks = context.riichi_stick_count ?? state.table_riichi_stick_count;
  tableCounterSummary.textContent = `本场 ${Number(honba || 0)}｜供托 ${Number(sticks || 0)}`;
  const confidence = Number(state.table_context_confidence || 0);
  tableContextStatus.textContent = `已连续两帧确认 · 置信度 ${(confidence * 100).toFixed(0)}%`;
  const budgetModel = strategy.risk_budget_model && typeof strategy.risk_budget_model === 'object'
    ? strategy.risk_budget_model
    : {};
  const placementAdjustment = Number(budgetModel.placement_adjustment || 0);
  const rewardBonus = Number(
    budgetModel.table_reward_bonus
    ?? context.win_reward_bonus
    ?? (Number(honba || 0) * 300 + Number(sticks || 0) * 1000)
  );
  const placementText = placementAdjustment
    ? `顺位让风险上限${placementAdjustment > 0 ? '+' : ''}${placementAdjustment.toFixed(0)}`
    : '当前分差不额外调整风险上限';
  tableStrategyImpact.textContent = `${placementText}；本场/供托共 ${rewardBonus.toLocaleString('zh-CN')} 点计入预计和牌收益。`;
}

function renderDashboard(data = {}) {
  const currentState = data.round_state || data.coach_state || data || {};
  const currentDecision = data.last_decision || data;
  const displaySnapshot = data.display_snapshot && typeof data.display_snapshot === 'object'
    ? data.display_snapshot
    : {};
  // Strategy UI and the native overlay intentionally consume the exact same
  // published snapshot. Current/quiet frames remain available to diagnostics.
  const state = displaySnapshot.round_state || currentState;
  const decision = displaySnapshot.last_decision || currentDecision;
  const live = data.live || {};
  const config = data.config || {};
  renderPreferences(data);
  if (riverTrackingModeInput && config.river_tracking_mode) {
    riverTrackingModeInput.value = config.river_tracking_mode;
  }
  setRecognitionMode(config.tile_recognition_mode || tileRecognitionModeInput?.value);
  if (settlementEnabledInput && typeof config.settlement_recognition_enabled === 'boolean') {
    settlementEnabledInput.checked = config.settlement_recognition_enabled;
  }
  if (settlementConfidenceInput && config.settlement_min_confidence !== undefined) {
    settlementConfidenceInput.value = String(config.settlement_min_confidence);
  }
  if (settlementFramesInput && config.settlement_confirm_frames !== undefined) {
    settlementFramesInput.value = String(config.settlement_confirm_frames);
  }
  if (settlementGapInput && config.settlement_confirm_max_gap_ms !== undefined) {
    settlementGapInput.value = String(config.settlement_confirm_max_gap_ms);
  }
  renderSettlementConfigSummary();
  const localPlan = state.local_direction || state.local_plan || state.current_plan || state.opening_plan || decision.suggestion;
  const localDetail = decision.detail || state.opening_plan || '';
  const overlayText = displaySnapshot.overlay_text || data.overlay_text || '';

  if (overlayText) {
    const lines = overlayText.split('\n');
    mainPlan.textContent = lines[0] || '等待手牌';
    planDetail.textContent = lines.slice(1).join('\n') || '还没有稳定手牌输入';
  } else {
    mainPlan.textContent = strategyHeadline(localPlan, listValues(state.target_shapes), '等待手牌');
    planDetail.textContent = strategyBrief(localPlan, localDetail, listValues(state.target_shapes), listValues(state.caution_points), '还没有稳定手牌输入');
  }
  analysisSource.textContent = '本地';
  analysisSource.classList.remove('is-ai');
  const style = state.play_style || 'riichi';
  const styleLabel = style === 'fast' ? '快攻' : '立直';
  biasValue.textContent = `${styleLabel} / ${compact(state.attack_defense_bias, 'neutral')}`;
  const displayStrategy = decision.perception?.strategy || decision.strategy || {};
  const posture = state.defense_posture || displayStrategy.posture || displayStrategy.mode || 'observe';
  defensePostureValue.textContent = ({ push: '推进', mawashi: '兜牌', fold: '全退' })[posture] || compact(posture, '观察');
  const riskBudget = state.defense_risk_budget ?? displayStrategy.risk_budget;
  riskBudgetValue.textContent = Number.isFinite(Number(riskBudget)) ? Number(riskBudget).toFixed(1) : '-';
  renderTableContext(state, displayStrategy);
  renderDefenseCandidates(displayStrategy);
  lastReason.textContent = compact(state.last_update_reason || decision.decision_type, '-');
  confidenceValue.textContent = percent(state.last_hand_confidence);
  updateCount.textContent = `${Number(state.update_count || 0)} updates`;
  decisionType.textContent = compact(decision.decision_type, 'observe');
  renderList(targetList, state.target_shapes, '暂无目标形状', 'tag');
  renderList(cautionList, state.caution_points, '暂无风险点', 'note');
  renderHand(firstNonEmptyList(state.last_hand_tiles, decision.hand_tiles, decision.perception?.hand?.hand_tiles));
  renderMelds(decision.perception?.meld?.melds || state.last_melds || []);
  renderOpponentMelds(
    firstNonEmptyPiles(
      state.last_opponent_melds,
      decision.perception?.river?.opponent_melds,
    ),
  );
  renderRiver(firstNonEmptyPiles(state.last_discard_piles, decision.perception?.river?.discard_piles));
  renderRoundArchive(data);
  renderPipeline(data);
  renderTimingLog(data.timing_log || []);
  decisionOutput.textContent = JSON.stringify(decision && Object.keys(decision).length ? decision : state, null, 2);
  renderLive(live);
  refreshFramePreview(live.last_frame_path).catch(() => {});
}

async function refreshStatus() {
  setStatus('刷新中');
  const data = await callPlugin('mahjong_coach_status', {}, 15000);
  renderDashboard(data);
  setStatus('ready');
}

function renderLive(live = {}) {
  const running = Boolean(live.running);
  const statusLabel = {
    waiting_for_game: '等待进入牌局',
    stopped: '已停止',
    starting: '启动中',
    observing: '识别中',
    waiting_for_window: '寻找雀魂窗口',
    view_obstructed: '牌桌被遮挡',
    verifying_new_round: '复核新局',
    settlement_candidate: '复核结算',
    round_settlement: '结算已确认',
    awaiting_next_round: '等待下一局',
    error: '运行异常',
  }[live.status];
  liveState.textContent = statusLabel || compact(live.status, '已停止');
  liveState.classList.toggle('is-running', running);
  liveFrame.textContent = `${Number(live.frame_index || 0)} frames`;
  liveWindow.textContent = compact(live.last_window_title || live.last_binding?.window_title, '未绑定窗口');
  liveError.textContent = compact(live.last_error || live.last_capture_source || live.last_frame_path, '-');
  syncLiveButtons(running);
  scheduleAutoRefresh(running);
}

function clearFramePreview(message = '等待实战截图') {
  lastPreviewPath = '';
  framePreview.removeAttribute('src');
  framePreview.hidden = true;
  framePreviewEmpty.hidden = false;
  framePreviewEmpty.textContent = message;
  tableRegionPreview.removeAttribute('src');
  tableRegionPreview.hidden = true;
  tableRegionPreviewEmpty.hidden = false;
  tableRegionPreviewEmpty.textContent = message;
  tableRegionPreviewState.textContent = message;
  settlementPreview.removeAttribute('src');
  settlementPreview.hidden = true;
  settlementPreviewEmpty.hidden = false;
  settlementPreviewEmpty.textContent = message;
  settlementPreviewState.textContent = message;
  framePreviewState.textContent = message;
}

async function refreshFramePreview(path) {
  const requestedPath = String(path || '').trim();
  framePreviewPath.textContent = requestedPath || '-';
  if (previewLoading) {
    // 加载期间只保留最新帧，当前请求完成后立刻补跑，避免结算图停在旧帧。
    // Keep only the newest queued frame while loading so settlement evidence cannot remain stale.
    queuedPreviewPath = requestedPath;
    if (!requestedPath) {
      clearFramePreview();
    }
    return;
  }
  if (!requestedPath) {
    clearFramePreview();
    return;
  }
  if (requestedPath === lastPreviewPath) {
    return;
  }
  previewLoading = true;
  queuedPreviewPath = null;
  framePreviewState.textContent = '加载截图';
  tableRegionPreviewState.textContent = '生成变换分区图';
  settlementPreviewState.textContent = '生成诊断图';
  try {
    const previewArgs = { image_path: requestedPath };
    const [frameResult, tableRegionResult, settlementResult] = await Promise.allSettled([
      callPlugin('mahjong_coach_frame_preview', previewArgs, 15000),
      callPlugin('mahjong_coach_table_region_preview', previewArgs, 15000),
      callPlugin('mahjong_coach_settlement_preview', previewArgs, 15000),
    ]);
    if (queuedPreviewPath !== null && queuedPreviewPath !== requestedPath) {
      return;
    }
    const data = frameResult.status === 'fulfilled' ? frameResult.value : {};
    if (!data.data_url) {
      clearFramePreview('截图不可用');
      return;
    }
    framePreview.src = data.data_url;
    framePreview.hidden = false;
    framePreviewEmpty.hidden = true;
    framePreviewPath.textContent = compact(data.image_path, requestedPath);
    framePreviewState.textContent = `${Number(data.width || 0)} × ${Number(data.height || 0)}`;
    const tableRegion = tableRegionResult.status === 'fulfilled' ? tableRegionResult.value : {};
    if (tableRegion.data_url && tableRegion.transformed) {
      tableRegionPreview.src = tableRegion.data_url;
      tableRegionPreview.hidden = false;
      tableRegionPreviewEmpty.hidden = true;
      tableRegionPreviewState.textContent = `${Number(tableRegion.width || 0)} × ${Number(tableRegion.height || 0)} · ${Number(tableRegion.detection_count || 0)} 框`;
    } else {
      tableRegionPreview.removeAttribute('src');
      tableRegionPreview.hidden = true;
      tableRegionPreviewEmpty.hidden = false;
      tableRegionPreviewEmpty.textContent = `牌桌变换不可用：${compact(tableRegion.reason, '未定位牌桌')}`;
      tableRegionPreviewState.textContent = '变换失败';
    }
    const diagnostic = settlementResult.status === 'fulfilled' ? settlementResult.value : {};
    if (diagnostic.data_url) {
      settlementPreview.src = diagnostic.data_url;
      settlementPreview.hidden = false;
      settlementPreviewEmpty.hidden = true;
      const kindLabel = {
        win: '和牌',
        exhaustive_draw: '荒牌流局',
        abortive_draw: '途中流局',
        unknown: '未分类结算',
      }[diagnostic.kind] || '未命中';
      settlementPreviewState.textContent = diagnostic.detected
        ? `${kindLabel} · ${percent(diagnostic.confidence)} · 当前帧`
        : `未命中 · ${compact(diagnostic.reason, '-')} · 当前帧`;
    } else {
      settlementPreview.removeAttribute('src');
      settlementPreview.hidden = true;
      settlementPreviewEmpty.hidden = false;
      settlementPreviewEmpty.textContent = '诊断图不可用';
      settlementPreviewState.textContent = '生成失败';
    }
    lastPreviewPath = String(data.image_path || requestedPath);
  } catch (_error) {
    clearFramePreview('预览加载失败');
  } finally {
    previewLoading = false;
    const nextPath = queuedPreviewPath;
    queuedPreviewPath = null;
    if (nextPath !== null && nextPath !== requestedPath) {
      if (nextPath) {
        refreshFramePreview(nextPath).catch(() => {});
      } else {
        clearFramePreview();
      }
    }
  }
}

function syncLiveButtons(running) {
  startLiveBtn.disabled = running;
  startYoloLiveBtn.disabled = running;
  stopLiveBtn.disabled = !running;
  [
    settlementEnabledInput,
    settlementConfidenceInput,
    settlementFramesInput,
    settlementGapInput,
  ].filter(Boolean).forEach((input) => {
    input.disabled = running;
  });
}

function scheduleAutoRefresh(running) {
  if (!running) {
    if (autoRefreshTimer) {
      window.clearInterval(autoRefreshTimer);
      autoRefreshTimer = 0;
    }
    return;
  }
  if (autoRefreshTimer) {
    return;
  }
  autoRefreshTimer = window.setInterval(() => {
    refreshStatus().catch((error) => {
      setStatus(error instanceof Error ? error.message : String(error));
    });
  }, 1200);
}

function keywordValues() {
  return String(keywordsInput.value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function tileValues(text) {
  return String(text || '')
    .split(/[,，、\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function resetRound() {
  setStatus('重置中');
  const data = await callPlugin('mahjong_coach_reset_round', { round_id: `round-${Date.now()}` }, 15000);
  renderDashboard(data);
  setStatus('ready');
}

async function analyzeFrame() {
  setStatus('分析中');
  const observedButtons = String(buttonsInput.value || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  const data = await callPlugin('mahjong_coach_analyze_frame', {
    image_path: imagePathInput.value.trim(),
    observed_buttons: observedButtons,
    self_turn_index: Number(turnInput.value || 0),
    force_checkpoint: Boolean(forceCheckpointInput.checked),
    ...settlementRuntimeArgs(),
    tile_recognition_mode: tileRecognitionModeInput ? tileRecognitionModeInput.value : 'legacy',
    round_wind: analysisRoundWindInput.value.trim(),
    seat_wind: analysisSeatWindInput.value.trim(),
    dora_tiles: tileValues(analysisDoraTilesInput.value),
  }, 30000);
  renderDashboard(data);
  setStatus(data.summary || 'ready');
}

function selectedProfileArgs() {
  return {
    rank: rankInput.value,
    room: roomInput.value,
    risk_tolerance: riskToleranceInput.value,
    goal_bias: goalBiasInput.value,
    call_bias: callBiasInput.value,
  };
}

async function loadWindowCandidates() {
  windowSelectionStatus.textContent = '正在扫描窗口…';
  const data = await callPlugin('mahjong_coach_window_candidates', { keywords: keywordValues() }, 15000);
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const matching = candidates.filter((item) => item.matches_keywords);
  const previous = windowCandidateSelect.value;
  windowCandidateSelect.replaceChildren(new Option('自动检测唯一窗口', ''));
  (matching.length ? matching : candidates).forEach((item) => {
    const size = item.width && item.height ? ` · ${item.width}×${item.height}` : '';
    const active = item.is_active ? ' · 当前' : '';
    windowCandidateSelect.add(new Option(`${item.title}${size}${active}`, item.title));
  });
  if ([...windowCandidateSelect.options].some((item) => item.value === previous)) {
    windowCandidateSelect.value = previous;
  }
  windowSelectionStatus.textContent = matching.length
    ? `找到 ${matching.length} 个雀魂窗口`
    : `找到 ${candidates.length} 个可见窗口`;
  return matching;
}

async function saveCapturePreferences() {
  const data = await callPlugin('mahjong_coach_save_preferences', {
    auto_start_live: Boolean(autoStartLiveInput.checked),
    target_window_title: windowCandidateSelect.value,
  }, 15000);
  preferencesHydrated = false;
  renderPreferences({ preferences: data.preferences }, true);
  windowSelectionStatus.textContent = '捕获设置已保存';
}

async function saveManualProfile() {
  const data = await callPlugin('mahjong_coach_save_profile', selectedProfileArgs(), 15000);
  preferencesHydrated = false;
  renderPreferences({ preferences: { profile: data.profile } }, true);
  profileLookupStatus.textContent = '手动画像已保存并生效';
}

async function searchPlayer() {
  const nickname = playerNicknameInput.value.trim();
  if (!nickname) throw new Error('请先输入雀魂昵称。');
  profileLookupStatus.textContent = '正在查询牌谱屋…';
  const data = await callPlugin('mahjong_coach_search_player', { nickname, limit: 10 }, 15000);
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  playerCandidateSelect.replaceChildren(new Option(candidates.length ? '请选择账号' : '没有候选账号', ''));
  candidates.forEach((item) => {
    const option = new Option(`${item.nickname} · ${item.rank || 'unknown'} · #${item.account_id}`, item.account_id);
    option.dataset.nickname = item.nickname || nickname;
    playerCandidateSelect.add(option);
  });
  profileLookupStatus.textContent = data.status === 'fallback_manual'
    ? `查询不可用，继续使用手动画像：${compact(data.error, '离线')}`
    : `找到 ${candidates.length} 个候选，请确认账号后应用`;
}

async function previewPlayer(forceRefresh = false) {
  const selected = playerCandidateSelect.selectedOptions[0];
  if (!selected?.value) return;
  profileLookupStatus.textContent = forceRefresh ? '正在刷新四麻统计…' : '正在读取画像建议…';
  const data = await callPlugin('mahjong_coach_preview_player_profile', {
    account_id: selected.value,
    nickname: selected.dataset.nickname || playerNicknameInput.value.trim(),
    force_refresh: Boolean(forceRefresh),
  }, 15000);
  if (data.status === 'fallback_manual') {
    profileLookupStatus.textContent = `统计不可用，继续使用手动画像：${compact(data.error, '离线')}`;
    return;
  }
  const profile = data.suggested_profile || {};
  setSelectValue(rankInput, profile.rank || 'unknown');
  setSelectValue(riskToleranceInput, profile.risk_tolerance || 'balanced');
  setSelectValue(goalBiasInput, profile.goal_bias || 'balanced');
  setSelectValue(callBiasInput, profile.call_bias || 'balanced');
  const rates = [
    profile.win_rate == null ? '' : `和牌 ${(profile.win_rate * 100).toFixed(1)}%`,
    profile.deal_in_rate == null ? '' : `放铳 ${(profile.deal_in_rate * 100).toFixed(1)}%`,
    profile.riichi_rate == null ? '' : `立直 ${(profile.riichi_rate * 100).toFixed(1)}%`,
    profile.call_rate == null ? '' : `副露 ${(profile.call_rate * 100).toFixed(1)}%`,
  ].filter(Boolean).join(' · ');
  profileSummary.textContent = `牌谱屋建议 · 待确认 · ${profile.risk_tolerance || 'balanced'}`;
  profileLookupStatus.textContent = `样本 ${Number(profile.sample_count || 0)} 局${rates ? ` · ${rates}` : ''}；可调整后确认`;
}

async function confirmPlayer() {
  const selected = playerCandidateSelect.selectedOptions[0];
  if (!selected?.value) throw new Error('请先选择要确认的玩家账号。');
  profileLookupStatus.textContent = '正在读取四麻统计…';
  const data = await callPlugin('mahjong_coach_confirm_player_profile', {
    account_id: selected.value,
    nickname: selected.dataset.nickname || playerNicknameInput.value.trim(),
    room: roomInput.value,
    risk_tolerance: riskToleranceInput.value,
    goal_bias: goalBiasInput.value,
    call_bias: callBiasInput.value,
  }, 15000);
  if (data.status === 'fallback_manual') {
    profileLookupStatus.textContent = `统计不可用，未更改画像：${compact(data.error, '离线')}`;
    return;
  }
  preferencesHydrated = false;
  renderPreferences({ preferences: { profile: data.profile } }, true);
  profileLookupStatus.textContent = `已确认并应用 ${data.profile?.nickname || '玩家'} 的建议画像`;
}

async function startLive() {
  setStatus('启动实战观察');
  const overlayRequested = Boolean(overlayInput.checked);
  const data = await callPlugin('mahjong_coach_start_live', {
    keywords: keywordValues(),
    interval_ms: Number(intervalInput.value || 400),
    overlay: overlayRequested,
    target_window_title: windowCandidateSelect.value,
    auto_start_live: Boolean(autoStartLiveInput.checked),
    ...settlementRuntimeArgs(),
    river_tracking_mode: riverTrackingModeInput ? riverTrackingModeInput.value : 'checkpoint',
    tile_recognition_mode: tileRecognitionModeInput ? tileRecognitionModeInput.value : 'legacy',
    round_wind: roundWindInput.value.trim(),
    seat_wind: seatWindInput.value.trim(),
    dora_tiles: tileValues(doraTilesInput.value),
  }, 15000);
  if (data.status === 'selection_required') {
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    windowCandidateSelect.replaceChildren(new Option('请选择目标窗口', ''));
    candidates.forEach((item) => windowCandidateSelect.add(new Option(item.title, item.title)));
    windowSelectionStatus.textContent = '检测到多个窗口，请选择后再次启动';
    setStatus('请选择雀魂窗口');
    return;
  }
  if (overlayRequested && data.overlay_ready !== true) {
    throw new Error('实战已请求启动，但操作浮窗未能打开，请查看插件日志。');
  }
  renderLive(data.live || {});
  await refreshStatus();
}

async function startYoloLive() {
  // 中文：快捷入口固定启用 YOLO26 和实时牌河，原版设置仍可随时切回。
  // English: The quick entry enables YOLO26 and live river tracking while keeping legacy selectable.
  setRecognitionMode('yolo26');
  riverTrackingModeInput.value = 'live';
  await startLive();
}

async function stopLive() {
  setStatus('停止实战观察');
  const data = await callPlugin('mahjong_coach_stop_live', {}, 15000);
  renderLive(data.live || {});
  await refreshStatus();
}

function bind(button, handler) {
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      await handler();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (button === startLiveBtn || button === startYoloLiveBtn || button === stopLiveBtn) {
        syncLiveButtons(liveState.classList.contains('is-running'));
      } else {
        button.disabled = false;
      }
    }
  });
}

bind(refreshBtn, refreshStatus);
bind(resetBtn, resetRound);
bind(analyzeBtn, analyzeFrame);
bind(startLiveBtn, startLive);
bind(startYoloLiveBtn, startYoloLive);
bind(stopLiveBtn, stopLive);
bind(refreshWindowsBtn, loadWindowCandidates);
bind(saveCapturePrefsBtn, saveCapturePreferences);
bind(saveProfileBtn, saveManualProfile);
bind(searchPlayerBtn, searchPlayer);
bind(refreshPlayerBtn, () => previewPlayer(true));
bind(confirmPlayerBtn, confirmPlayer);

playerCandidateSelect.addEventListener('change', () => {
  previewPlayer(false).catch((error) => {
    profileLookupStatus.textContent = error instanceof Error ? error.message : String(error);
  });
});

legacyModeBtn.addEventListener('click', () => setRecognitionMode('legacy'));
yoloModeBtn.addEventListener('click', () => setRecognitionMode('yolo26'));
[
  settlementEnabledInput,
  settlementConfidenceInput,
  settlementFramesInput,
  settlementGapInput,
].filter(Boolean).forEach((input) => {
  input.addEventListener('input', renderSettlementConfigSummary);
  input.addEventListener('change', renderSettlementConfigSummary);
});

renderSettlementConfigSummary();
refreshStatus().catch((error) => {
  setStatus(error instanceof Error ? error.message : String(error));
});
