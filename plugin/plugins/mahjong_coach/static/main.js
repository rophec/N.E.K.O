const PLUGIN_ID = 'mahjong_coach';
const RUNS_URL = '/runs';
const RUN_TIMEOUT_MS = 30000;

const statusLine = document.getElementById('statusLine');
const refreshBtn = document.getElementById('refreshBtn');
const resetBtn = document.getElementById('resetBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const startLiveBtn = document.getElementById('startLiveBtn');
const stopLiveBtn = document.getElementById('stopLiveBtn');
const imagePathInput = document.getElementById('imagePathInput');
const turnInput = document.getElementById('turnInput');
const buttonsInput = document.getElementById('buttonsInput');
const forceCheckpointInput = document.getElementById('forceCheckpointInput');
const keywordsInput = document.getElementById('keywordsInput');
const intervalInput = document.getElementById('intervalInput');
const overlayInput = document.getElementById('overlayInput');
const riverTrackingModeInput = document.getElementById('riverTrackingModeInput');
const roundWindInput = document.getElementById('roundWindInput');
const seatWindInput = document.getElementById('seatWindInput');
const doraTilesInput = document.getElementById('doraTilesInput');
const analysisRoundWindInput = document.getElementById('analysisRoundWindInput');
const analysisSeatWindInput = document.getElementById('analysisSeatWindInput');
const analysisDoraTilesInput = document.getElementById('analysisDoraTilesInput');
const analysisSource = document.getElementById('analysisSource');
const mainPlan = document.getElementById('mainPlan');
const planDetail = document.getElementById('planDetail');
const biasValue = document.getElementById('biasValue');
const lastReason = document.getElementById('lastReason');
const confidenceValue = document.getElementById('confidenceValue');
const updateCount = document.getElementById('updateCount');
const targetList = document.getElementById('targetList');
const cautionList = document.getElementById('cautionList');
const handTiles = document.getElementById('handTiles');
const handCount = document.getElementById('handCount');
const riverTiles = document.getElementById('riverTiles');
const riverCount = document.getElementById('riverCount');
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
let autoRefreshTimer = 0;

function setStatus(text) {
  statusLine.textContent = text || '';
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

function renderHand(tiles) {
  handTiles.replaceChildren();
  const values = Array.isArray(tiles) ? tiles.filter(Boolean) : [];
  handCount.textContent = `${values.length} 张`;
  if (!values.length) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = '暂无手牌';
    handTiles.appendChild(empty);
    return;
  }
  values.forEach((tile) => {
    const node = document.createElement('span');
    node.className = `tile tile-${String(tile).slice(-1)}`;
    node.textContent = String(tile);
    handTiles.appendChild(node);
  });
}

function renderRiver(piles = {}) {
  riverTiles.replaceChildren();
  const entries = Object.entries(piles || {});
  const total = entries.reduce((count, [, items]) => count + (Array.isArray(items) ? items.length : 0), 0);
  riverCount.textContent = `${total} 张`;
  if (!total) {
    const empty = document.createElement('span');
    empty.className = 'empty-text';
    empty.textContent = '暂无牌河';
    riverTiles.appendChild(empty);
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
      row.appendChild(node);
    });
    riverTiles.appendChild(row);
  });
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
  if (live.status === 'error') {
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

  return [
    pipelineStep('capture', '截图/窗口', captureStatus, captureDetail),
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
    empty.colSpan = 15;
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
      compact(item.river_mode),
      compact(item.river_reason),
      formatMs(item.locate_ms),
      formatMs(item.capture_ms),
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
      if (index >= 6) {
        cell.className = 'timing-number';
      }
      row.appendChild(cell);
    });
    timingLogBody.appendChild(row);
  });
}

function renderDashboard(data = {}) {
  const state = data.round_state || data.coach_state || data || {};
  const decision = data.last_decision || data;
  const live = data.live || {};
  const config = data.config || {};
  if (riverTrackingModeInput && config.river_tracking_mode) {
    riverTrackingModeInput.value = config.river_tracking_mode;
  }
  const localPlan = state.local_direction || state.local_plan || state.current_plan || state.opening_plan || decision.suggestion;
  const localDetail = decision.detail || state.opening_plan || '';
  const overlayText = data.overlay_text || '';

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
  lastReason.textContent = compact(state.last_update_reason || decision.decision_type, '-');
  confidenceValue.textContent = percent(state.last_hand_confidence);
  updateCount.textContent = `${Number(state.update_count || 0)} updates`;
  decisionType.textContent = compact(decision.decision_type, 'observe');
  renderList(targetList, state.target_shapes, '暂无目标形状', 'tag');
  renderList(cautionList, state.caution_points, '暂无风险点', 'note');
  renderHand(firstNonEmptyList(state.last_hand_tiles, decision.hand_tiles, decision.perception?.hand?.hand_tiles));
  renderRiver(firstNonEmptyPiles(state.last_discard_piles, decision.perception?.river?.discard_piles));
  renderPipeline(data);
  renderTimingLog(data.timing_log || []);
  decisionOutput.textContent = JSON.stringify(decision && Object.keys(decision).length ? decision : state, null, 2);
  renderLive(live);
}

async function refreshStatus() {
  setStatus('刷新中');
  const data = await callPlugin('mahjong_coach_status', {}, 15000);
  renderDashboard(data);
  setStatus('ready');
}

function renderLive(live = {}) {
  const running = Boolean(live.running);
  liveState.textContent = compact(live.status, 'stopped');
  liveState.classList.toggle('is-running', running);
  liveFrame.textContent = `${Number(live.frame_index || 0)} frames`;
  liveWindow.textContent = compact(live.last_window_title || live.last_binding?.window_title, '未绑定窗口');
  liveError.textContent = compact(live.last_error || live.last_capture_source || live.last_frame_path, '-');
  startLiveBtn.disabled = running;
  stopLiveBtn.disabled = !running;
  scheduleAutoRefresh(running);
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
  renderDashboard(data.round_state || data);
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
    round_wind: analysisRoundWindInput.value.trim(),
    seat_wind: analysisSeatWindInput.value.trim(),
    dora_tiles: tileValues(analysisDoraTilesInput.value),
  }, 30000);
  renderDashboard(data);
  setStatus(data.summary || 'ready');
}

async function startLive() {
  setStatus('启动实战观察');
  const data = await callPlugin('mahjong_coach_start_live', {
    keywords: keywordValues(),
    interval_ms: Number(intervalInput.value || 1200),
    overlay: Boolean(overlayInput.checked),
    river_tracking_mode: riverTrackingModeInput ? riverTrackingModeInput.value : 'checkpoint',
    round_wind: roundWindInput.value.trim(),
    seat_wind: seatWindInput.value.trim(),
    dora_tiles: tileValues(doraTilesInput.value),
  }, 15000);
  renderLive(data.live || {});
  await refreshStatus();
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
      button.disabled = false;
    }
  });
}

bind(refreshBtn, refreshStatus);
bind(resetBtn, resetRound);
bind(analyzeBtn, analyzeFrame);
bind(startLiveBtn, startLive);
bind(stopLiveBtn, stopLive);

refreshStatus().catch((error) => {
  setStatus(error instanceof Error ? error.message : String(error));
});
