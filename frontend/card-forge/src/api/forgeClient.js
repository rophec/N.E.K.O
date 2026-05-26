// forge_server (port 3002) HTTP 客户端。
// dev: vite proxy /forge → http://localhost:3002
// IIFE 内嵌主应用时：通过 window.NEKO_FORGE_API_BASE（Jinja 注入）走 same-origin 或反代。

const baseUrl = () => {
  if (typeof window !== 'undefined' && window.NEKO_FORGE_API_BASE) {
    return String(window.NEKO_FORGE_API_BASE).replace(/\/+$/, '')
  }
  return ''
}

function url(path) {
  return `${baseUrl()}${path}`
}

export async function fetchForgeFacts({ character, excludeFactIds = [], excludeHashes = [], limit = 5 } = {}) {
  if (!character) {
    return {
      facts: [],
      character: '',
      fallbackReason: 'runtime_character_hint_missing',
      error: 'active_neko_runtime_not_linked',
    }
  }
  const qs = new URLSearchParams()
  qs.set('runtime_character_hint', character)
  qs.set('include_absorbed', 'true')
  qs.set('min_importance', '0')
  qs.set('limit', String(limit))
  const ids = Array.from(new Set(excludeFactIds.filter(Boolean)))
  const hashes = Array.from(new Set(excludeHashes.filter(Boolean)))
  if (ids.length) qs.set('exclude_fact_ids', ids.join(','))
  if (hashes.length) qs.set('exclude_hashes', hashes.join(','))
  const res = await fetch(url(`/forge/facts?${qs.toString()}`))
  if (!res.ok) throw new Error(`forge-facts http ${res.status}`)
  return res.json()
}

export async function requestForgeStory(body) {
  const res = await fetch(url('/forge/card-story'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok || !data?.success || !data?.story) return null
  return data
}

export async function listForgedInventory(character) {
  if (!character) return []
  const res = await fetch(url(`/forge/inventory?character=${encodeURIComponent(character)}`))
  if (!res.ok) return []
  const data = await res.json().catch(() => ({}))
  return Array.isArray(data?.cards) ? data.cards : []
}

export async function addForgedInventory(character, card) {
  if (!character || !card?.id) throw new Error('character_and_card_required')
  const res = await fetch(url('/forge/inventory'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ character, card }),
  })
  if (!res.ok) throw new Error(`forge-inventory POST http ${res.status}`)
  return res.json()
}

export async function deleteForgedInventory(character, cardId) {
  if (!character || !cardId) return false
  const res = await fetch(url(`/forge/inventory/${encodeURIComponent(cardId)}?character=${encodeURIComponent(character)}`), {
    method: 'DELETE',
  })
  if (!res.ok) return false
  const data = await res.json().catch(() => ({}))
  return Boolean(data?.removed)
}

// 获取当前 NEKO 主应用绑定的猫娘名（走 /battle-arena/avatar/left endpoint，
// dev 通过 vite proxy 转到 48911；IIFE 模式同源）。
export async function fetchActiveCharacterName() {
  try {
    const res = await fetch('/battle-arena/avatar/left')
    if (!res.ok) return ''
    const data = await res.json().catch(() => ({}))
    return typeof data?.name === 'string' ? data.name : ''
  } catch {
    return ''
  }
}
