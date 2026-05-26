// 卡牌铸造数据模型（card-forge 完整版）。
// 这里只描述"故事卡"本身的形状——卡名、性格气质、故事文本、来源记忆引用。
// 不携带任何战斗规则（cost / type / 主效果 / Combo / 稀有度 / 基础卡映射）；
// 那些是上层（如猫娘大乱斗）消费这些卡时自行决定怎么"赋值"的事情。

// 性格气质池：影响 LLM 写故事时的语气、动作选择。不是战斗属性。
export const FORGE_PERSONA_ATTRS = [
  { id: 'passion', name: '热情' },
  { id: 'gentle', name: '温柔' },
  { id: 'cool', name: '高冷' },
  { id: 'natural', name: '天然' },
]

const personaNameById = (id) => FORGE_PERSONA_ATTRS.find(attr => attr.id === id)?.name || id

// 临时事件池：当 NEKO 当前猫娘还没真实 facts 可铸造时，给前端一些占位事件
// 用来跑通流程；不会替代真实记忆系统。
const TEMP_FORGED_CARD_EVENTS = [
  { name: '临时练习室事件', storyLead: '午后练习室里，猫娘把一次差点失败的配合记成了新的灵感。' },
  { name: '临时便利店事件', storyLead: '深夜便利店门口，一句没说出口的鼓励被悄悄记下。' },
  { name: '临时屋檐事件', storyLead: '雨后的屋檐下，短暂的并肩等待让记忆多了一层默契。' },
  { name: '临时贩卖机事件', storyLead: '自动贩卖机前的最后一罐饮料被当作一个小小的约定保存下来。' },
  { name: '临时地铁站事件', storyLead: '走错路的地铁站里，绕远的时间反而给了这段记忆新的形状。' },
  { name: '临时手电光事件', storyLead: '停电时借来的手电光，把普通回忆照成了值得记下的奇遇。' },
]

function pickRandom(list) {
  return list[Math.floor(Math.random() * list.length)]
}

function getEventStoryLead(event = {}) {
  if (typeof event.storyLead === 'string' && event.storyLead.trim()) return event.storyLead.trim()
  if (typeof event.factText === 'string' && event.factText.trim()) return event.factText.trim()
  if (typeof event.text === 'string' && event.text.trim()) return event.text.trim()
  if (typeof event.summary === 'string' && event.summary.trim()) return event.summary.trim()
  return pickRandom(TEMP_FORGED_CARD_EVENTS).storyLead
}

function getEventName(event = {}, storyLead = '') {
  if (typeof event.name === 'string' && event.name.trim()) return event.name.trim()
  if (typeof event.sourceEventName === 'string' && event.sourceEventName.trim()) return event.sourceEventName.trim()
  if (storyLead) return storyLead.length > 24 ? `${storyLead.slice(0, 24)}…` : storyLead
  return pickRandom(TEMP_FORGED_CARD_EVENTS).name
}

function buildTemporaryForgedStory(storyLead, card = {}) {
  const lead = storyLead || '这段记忆暂时还没有完整记录。'
  const attrName = card.attrName || '未确认气质'
  return [
    `${lead}`,
    `猫娘会以 ${attrName} 的性格气质重新看待这段记忆，让故事继续沿着原本的情绪发展。`,
    `正式故事生成接入成功后，只会参考故事引子和性格气质，不参考任何其他规则。`,
    '【临时前端占位】后续接入正式故事生成后，请用真实生成结果替换本段。',
  ].join('\n')
}

export function composeForgedCardStory(storyLead, generatedStory, card = {}) {
  const lead = storyLead || ''
  const story = typeof generatedStory === 'string' ? generatedStory.trim() : ''
  if (!story) return buildTemporaryForgedStory(lead, card)
  return story
}

/**
 * 铸造一张纯故事卡。
 * 唯一会"被赋值"的随机量是性格气质（影响 LLM 写作时的语气）。
 * 不携带任何游戏规则字段；上层（如猫娘大乱斗）需要时自行映射。
 */
export function createForgedBrawlCard(event = {}, options = {}) {
  const persona = options.personaId
    ? FORGE_PERSONA_ATTRS.find(attr => attr.id === options.personaId) || pickRandom(FORGE_PERSONA_ATTRS)
    : pickRandom(FORGE_PERSONA_ATTRS)
  const id = `forged-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const storyLead = getEventStoryLead(event)
  const sourceEventName = getEventName(event, storyLead)
  const story = composeForgedCardStory(storyLead, event.story || event.generatedStory, { attrName: persona.name })

  return {
    id,
    forged: true,
    name: sourceEventName,
    attrId: persona.id,
    attrName: personaNameById(persona.id),
    story,
    summary: story,
    storyLead,
    sourceFactId: event.sourceFactId || event.factId || null,
    sourceFactHash: event.sourceFactHash || event.factHash || null,
    sourceCharacter: event.sourceCharacter || null,
    sourceKind: event.sourceKind || (event.sourceFactId || event.factId ? 'fact' : 'temporary'),
    sourceEventName,
    storyGenerationStatus: event.story || event.generatedStory ? 'ready' : 'pending-llm',
    forgedAt: Date.now(),
  }
}
