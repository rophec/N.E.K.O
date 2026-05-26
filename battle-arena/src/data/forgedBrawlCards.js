// battle-arena 端是铸造卡的"只读消费者"。
// 写入（createForgedBrawlCard）只在 frontend/card-forge 里发生；
// 这里只保留组卡/战斗需要的 schema 常量、normalize 工具，以及把 inventory load/save/delete
// 桥接到 forge_server (port 3002) 的 HTTP API。

export const FORGED_BRAWL_CARDS_STORAGE_KEY = 'neko-brawl-forged-cards'

export const BRAWL_ATTRS = [
  { id: 'passion', name: '热情' },
  { id: 'gentle', name: '温柔' },
  { id: 'cool', name: '高冷' },
  { id: 'natural', name: '天然' },
]

const attrNameById = (id) => BRAWL_ATTRS.find(attr => attr.id === id)?.name || id

export const BRAWL_CARD_EFFECT_POOL = [
  { code: 'C001', name: '午后扑抱', attrId: 'passion', cost: 1, type: '攻击', mainText: '对Boss造成1点伤害', comboText: '额外造成1点伤害', main: { damage: 1 }, combo: { damage: 1 } },
  { code: 'C002', name: '亮晶晶眼神', attrId: 'gentle', cost: 1, type: '回复', mainText: '回复生命最低的己方玩家1点生命', comboText: '自身回复1点生命', main: { healLowest: 1 }, combo: { healSelf: 1 } },
  { code: 'C003', name: '尾巴在说话', attrId: 'cool', cost: 1, type: '防御', mainText: '为自己获得1点护盾', comboText: '为队友提供1点护盾', main: { shieldSelf: 1 }, combo: { shieldOther: 1 } },
  { code: 'C004', name: '云朵经过的三秒', attrId: 'natural', cost: 1, type: '抽牌', mainText: '抽1张牌', comboText: '额外抽1张牌', main: { draw: 1 }, combo: { draw: 1 } },
  { code: 'C005', name: '还没认输呢', attrId: 'passion', cost: 2, type: '攻击', mainText: '对Boss造成2点伤害', comboText: '额外造成1点伤害', main: { damage: 2 }, combo: { damage: 1 } },
  { code: 'C006', name: '怀中心跳', attrId: 'cool', cost: 2, type: '防御', mainText: '本回合Boss对自己造成的伤害-2', comboText: '队友本回合受到的伤害-2', main: { reduceSelfDamageThisRound: 2 }, combo: { reduceOtherDamageThisRound: 2 } },
  { code: 'C007', name: '熬夜到头秃', attrId: 'cool', cost: 2, type: '强化', mainText: '下回合造成伤害+2', comboText: '获得2点护盾', main: { damageBonusNext: 2 }, combo: { shieldSelf: 2 } },
  { code: 'C008', name: '拂面微风', attrId: 'natural', cost: 2, type: '回复', mainText: '双方玩家各回复1点生命', comboText: '额外为双方各获得1点护盾', main: { healBoth: 1 }, combo: { shieldBoth: 1 } },
  { code: 'C009', name: '纸箱里的秘密计划', attrId: 'gentle', cost: 2, type: '控制', mainText: '对Boss造成1点伤害，并使Boss下次攻击伤害-1', comboText: '额外造成1点伤害', main: { damage: 1, bossDamageReductionNext: 1 }, combo: { damage: 1 } },
  { code: 'C010', name: '屋顶上的晚安', attrId: 'cool', cost: 3, type: '回复', mainText: '回复双方玩家各2点生命', comboText: '清除1个负面状态', main: { healBoth: 2 }, combo: { clearDebuff: 1 } },
  { code: 'C011', name: '生人勿近', attrId: 'natural', cost: 3, type: '防御', mainText: '对Boss造成2点伤害，并为双方各获得1点护盾', comboText: '本回合Boss造成伤害-1', main: { damage: 2, shieldBoth: 1 }, combo: { bossDamageReductionThisRound: 1 } },
  { code: 'C012', name: '用尽全力奔向你', attrId: 'gentle', cost: 3, type: '攻击', mainText: '对Boss造成4点伤害', comboText: '额外造成2点伤害', main: { damage: 4 }, combo: { damage: 2 } },
  { code: 'C013', name: '完全⭐奇迹', attrId: 'passion', cost: 4, type: '控制', mainText: '对Boss造成3点伤害，并封锁boss下回合行动', comboText: '自身获得2点护盾', main: { damage: 3, skipBossNext: true }, combo: { shieldSelf: 2 } },
]

function pickRandom(list) {
  return list[Math.floor(Math.random() * list.length)]
}

function normalizeEffect(effect = {}) {
  return { ...effect }
}

function buildTemporaryForgedStory(storyLead, card = {}) {
  const lead = storyLead || '这段记忆暂时还没有完整记录。'
  const attrName = card.attrName || '未确认属性'
  return [
    `${lead}`,
    `猫娘会以 ${attrName} 的性格气质重新看待这段记忆，让故事继续沿着原本的情绪发展。`,
    `正式故事生成接入成功后，只会参考故事引子和主属性性格，不参考卡名、费用、类型、效果或任何游戏规则。`,
    '【临时前端占位】后续接入正式故事生成后，请用真实生成结果替换本段，并继续让故事自然包含原本的事件内容。',
  ].join('\n')
}

export function composeForgedCardStory(storyLead, generatedStory, card = {}) {
  const lead = storyLead || ''
  const story = typeof generatedStory === 'string' ? generatedStory.trim() : ''
  if (!story) return buildTemporaryForgedStory(lead, card)
  return story
}

export function normalizeForgedBrawlCard(card) {
  if (!card || typeof card !== 'object') return null
  const base = BRAWL_CARD_EFFECT_POOL.find(item => item.code === card.baseCode)
    || BRAWL_CARD_EFFECT_POOL.find(item => item.code === card.code)
    || BRAWL_CARD_EFFECT_POOL[0]
  const comboAttrId = BRAWL_ATTRS.some(attr => attr.id === card.comboAttrId)
    ? card.comboAttrId
    : pickRandom(BRAWL_ATTRS).id

  const storyLead = card.storyLead || card.factText || card.text || card.eventLead || card.summary || ''
  const story = composeForgedCardStory(storyLead, card.story, base)

  return {
    ...card,
    id: card.id || `forged-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    code: card.code || `${base.code}-F-${Math.random().toString(36).slice(2, 8)}`,
    baseCode: card.baseCode || base.code,
    forged: true,
    name: card.name || `${base.name}(Forged)`,
    title: card.title || card.name || `${base.name}(Forged)`,
    attrId: base.attrId,
    attrName: attrNameById(base.attrId),
    comboAttrId,
    comboAttrName: attrNameById(comboAttrId),
    cost: base.cost,
    type: base.type,
    mainText: base.mainText,
    comboText: base.comboText,
    main: normalizeEffect(base.main),
    combo: normalizeEffect(base.combo),
    story,
    summary: card.summary || story,
    storyLead,
    sourceFactId: card.sourceFactId || card.factId || null,
    sourceFactHash: card.sourceFactHash || card.factHash || null,
    sourceCharacter: card.sourceCharacter || null,
    sourceKind: card.sourceKind || (card.sourceFactId || card.factId ? 'fact' : 'temporary'),
    sourceEventName: card.sourceEventName || '临时奇遇事件',
    storyGenerationStatus: card.storyGenerationStatus || (card.story ? 'ready' : 'pending-llm'),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP 桥接：forge_server (port 3002)
// ─────────────────────────────────────────────────────────────────────────────

const forgeBase = () => {
  if (typeof window !== 'undefined' && window.NEKO_FORGE_API_BASE) {
    return String(window.NEKO_FORGE_API_BASE).replace(/\/+$/, '')
  }
  // dev: vite proxy /forge → 3002（battle-arena vite.config.js 已配置）
  return ''
}

/** 异步从 forge_server 拉当前猫娘的铸造卡仓库。 */
export async function loadForgedBrawlCards(character) {
  if (typeof window === 'undefined' || !character) return []
  try {
    const res = await fetch(`${forgeBase()}/forge/inventory?character=${encodeURIComponent(character)}`)
    if (!res.ok) return []
    const data = await res.json().catch(() => ({}))
    const cards = Array.isArray(data?.cards) ? data.cards : []
    return cards.map(normalizeForgedBrawlCard).filter(Boolean)
  } catch {
    return []
  }
}

/**
 * 兼容旧调用：异步删除一张铸造卡并返回最新仓库。
 * 调用方需要传入 character；不传时只本地过滤当前列表（不落盘）。
 */
export async function deleteForgedBrawlCard(cardRef, character) {
  if (!cardRef?.id || !character) return loadForgedBrawlCards(character)
  try {
    await fetch(`${forgeBase()}/forge/inventory/${encodeURIComponent(cardRef.id)}?character=${encodeURIComponent(character)}`, {
      method: 'DELETE',
    })
  } catch {
    // 网络失败时仍重新拉一次仓库，UI 自洽
  }
  return loadForgedBrawlCards(character)
}
