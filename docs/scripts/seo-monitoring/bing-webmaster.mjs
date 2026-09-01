import { readFile } from 'node:fs/promises'

const API_BASE = 'https://ssl.bing.com/webmaster/api.svc/json'
const METHODS = [
  'GetRankAndTrafficStats',
  'GetQueryStats',
  'GetPageStats',
  'GetCrawlStats',
  'GetFeeds',
]

function finiteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function searchPosition(value) {
  const number = finiteNumber(value)
  return number != null && number >= 0 ? number : null
}

function dayValue(value) {
  const match = String(value ?? '').match(/^\/Date\((-?\d+)/u)
  if (match) return new Date(Number(match[1]))
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

export function parseBingDate(value) {
  const date = dayValue(value)
  return date ? date.toISOString().slice(0, 10) : null
}

export async function readBingWebmasterApiKey(path) {
  const value = (await readFile(path, 'utf8')).trim()
  if (!value) throw new TypeError('Bing Webmaster API key file is empty')
  return value
}

function addDays(date, days) {
  const result = new Date(`${date}T00:00:00.000Z`)
  result.setUTCDate(result.getUTCDate() + days)
  return result.toISOString().slice(0, 10)
}

function sumRows(rows) {
  return rows.reduce((total, row) => ({
    clicks: total.clicks + (row.clicks ?? 0),
    impressions: total.impressions + (row.impressions ?? 0),
  }), { clicks: 0, impressions: 0 })
}

function trend(current, previous) {
  return {
    clicks: current.clicks - previous.clicks,
    impressions: current.impressions - previous.impressions,
  }
}

function trafficSummary(rows) {
  const daily = rows.map(row => ({
    date: parseBingDate(row.Date),
    clicks: finiteNumber(row.Clicks) ?? 0,
    impressions: finiteNumber(row.Impressions) ?? 0,
  })).filter(row => row.date).sort((left, right) => left.date.localeCompare(right.date))
  const dataThrough = daily.at(-1)?.date ?? null
  if (!dataThrough) {
    return { frequency: 'daily', dataThrough: null, latestDay: null, windows: null, recent7: null, previous7: null, trend7: null, daily: [] }
  }
  const recentStart = addDays(dataThrough, -6)
  const previousEnd = addDays(recentStart, -1)
  const previousStart = addDays(previousEnd, -6)
  const recent7 = sumRows(daily.filter(row => row.date >= recentStart && row.date <= dataThrough))
  const previous7 = sumRows(daily.filter(row => row.date >= previousStart && row.date <= previousEnd))
  return {
    frequency: 'daily',
    dataThrough,
    latestDay: daily.at(-1),
    windows: {
      recent7: { startDate: recentStart, endDate: dataThrough },
      previous7: { startDate: previousStart, endDate: previousEnd },
    },
    recent7,
    previous7,
    trend7: trend(recent7, previous7),
    daily: daily.slice(-35),
  }
}

function latestWeeklyRows(rows, valueName) {
  const normalized = rows.map(row => ({
    value: String(row.Query ?? '').trim(),
    date: parseBingDate(row.Date),
    clicks: finiteNumber(row.Clicks) ?? 0,
    impressions: finiteNumber(row.Impressions) ?? 0,
    averageClickPosition: searchPosition(row.AvgClickPosition),
    averageImpressionPosition: searchPosition(row.AvgImpressionPosition),
  })).filter(row => row.value && row.date)
  const dataThrough = normalized.reduce((latest, row) => row.date > latest ? row.date : latest, '') || null
  const items = dataThrough
    ? normalized.filter(row => row.date === dataThrough)
      .sort((left, right) => right.clicks - left.clicks || right.impressions - left.impressions || left.value.localeCompare(right.value))
      .slice(0, 25)
      .map(row => ({ [valueName]: row.value, ...Object.fromEntries(Object.entries(row).filter(([key]) => key !== 'value')) }))
    : []
  return { frequency: 'weekly', dataThrough, items }
}

function crawlSummary(rows) {
  const normalized = rows.map(row => ({
    date: parseBingDate(row.Date),
    crawledPages: finiteNumber(row.CrawledPages),
    inIndex: finiteNumber(row.InIndex),
    crawlErrors: finiteNumber(row.CrawlErrors),
    code2xx: finiteNumber(row.Code2xx),
    code4xx: finiteNumber(row.Code4xx),
    code5xx: finiteNumber(row.Code5xx),
    blockedByRobotsTxt: finiteNumber(row.BlockedByRobotsTxt),
    dnsFailures: finiteNumber(row.DnsFailures),
    connectionTimeouts: finiteNumber(row.ConnectionTimeouts),
    containsMalware: finiteNumber(row.ContainsMalware),
  })).filter(row => row.date).sort((left, right) => left.date.localeCompare(right.date))
  return { frequency: 'daily', dataThrough: normalized.at(-1)?.date ?? null, latestDay: normalized.at(-1) ?? null }
}

function feedSummary(rows) {
  return rows.map(row => ({
    url: row.Url ?? row.FeedUrl ?? null,
    status: row.Status ?? null,
    urlCount: finiteNumber(row.UrlCount),
    lastCrawled: parseBingDate(row.LastCrawled),
    lastSubmitted: parseBingDate(row.LastSubmitted),
  })).filter(row => row.url)
}

function responseRows(payload) {
  if (Array.isArray(payload?.d)) return payload.d
  if (Array.isArray(payload?.d?.results)) return payload.d.results
  if (Array.isArray(payload)) return payload
  return []
}

function safeReason(error, apiKey) {
  const message = String(error?.message ?? 'unknown Bing Webmaster API error')
  return apiKey
    ? message.replaceAll(apiKey, '[REDACTED]').replaceAll(encodeURIComponent(apiKey), '[REDACTED]')
    : message
}

async function request(method, siteUrl, { apiKey, fetchImpl, timeoutMs }) {
  const url = new URL(`${API_BASE}/${method}`)
  if (siteUrl) url.searchParams.set('siteUrl', siteUrl)
  url.searchParams.set('apikey', apiKey)
  let response
  try {
    response = await fetchImpl(url, { signal: AbortSignal.timeout(timeoutMs) })
  } catch (error) {
    throw new Error(`${method} request failed: ${safeReason(error, apiKey)}`)
  }
  if (!response.ok) throw new Error(`${method} returned HTTP ${response.status}`)
  try {
    return responseRows(await response.json())
  } catch {
    throw new Error(`${method} returned invalid JSON`)
  }
}

function normalizedSiteUrl(value) {
  try {
    const url = new URL(value)
    url.hash = ''
    url.search = ''
    url.pathname = url.pathname.replace(/\/+$/u, '') || '/'
    return url.href.toLowerCase()
  } catch {
    return String(value ?? '').trim().replace(/\/+$/u, '').toLowerCase()
  }
}

function emptyResult(definition, collectedAt, { status, availability, reason, isVerified }) {
  return {
    id: definition.id,
    label: definition.label,
    siteUrl: definition.siteUrl,
    status,
    availability,
    reason,
    isVerified,
    collectedAt,
    successfulMethods: 0,
    totalMethods: METHODS.length,
    traffic: trafficSummary([]),
    queries: latestWeeklyRows([], 'query'),
    pages: latestWeeklyRows([], 'page'),
    crawl: crawlSummary([]),
    feeds: [],
    errors: [],
  }
}

export async function collectBingWebmasterSite(definition, {
  apiKey,
  fetchImpl = fetch,
  timeoutMs = 20_000,
} = {}) {
  if (!apiKey) throw new TypeError('Bing Webmaster API key is required')
  const collectedAt = new Date().toISOString()
  const userSites = await request('GetUserSites', null, { apiKey, fetchImpl, timeoutMs })
  const accountSite = userSites.find(site => normalizedSiteUrl(site.Url) === normalizedSiteUrl(definition.siteUrl))
  if (!accountSite) {
    return emptyResult(definition, collectedAt, {
      status: 'not_added',
      availability: 'not_added',
      reason: 'Site is not present in the Bing Webmaster account',
      isVerified: false,
    })
  }
  if (accountSite.IsVerified !== true) {
    return emptyResult(definition, collectedAt, {
      status: 'not_verified',
      availability: 'not_verified',
      reason: 'Site is present in Bing Webmaster but ownership is not verified',
      isVerified: false,
    })
  }
  const settled = await Promise.allSettled(METHODS.map(method => request(method, definition.siteUrl, {
    apiKey,
    fetchImpl,
    timeoutMs,
  })))
  const byMethod = Object.fromEntries(METHODS.map((method, index) => [method, settled[index]]))
  const errors = METHODS.flatMap(method => byMethod[method].status === 'rejected'
    ? [{ method, reason: safeReason(byMethod[method].reason, apiKey) }]
    : [])
  const rows = method => byMethod[method].status === 'fulfilled' ? byMethod[method].value : []
  const traffic = trafficSummary(rows('GetRankAndTrafficStats'))
  const queries = latestWeeklyRows(rows('GetQueryStats'), 'query')
  const pages = latestWeeklyRows(rows('GetPageStats'), 'page')
  const crawl = crawlSummary(rows('GetCrawlStats'))
  const feeds = feedSummary(rows('GetFeeds'))
  const successfulMethods = METHODS.length - errors.length
  const hasData = Boolean(traffic.latestDay || queries.items.length || pages.items.length || crawl.latestDay || feeds.length)
  return {
    id: definition.id,
    label: definition.label,
    siteUrl: definition.siteUrl,
    isVerified: true,
    status: successfulMethods === 0 ? 'unavailable' : errors.length > 0 ? 'partial' : 'ok',
    availability: hasData ? 'available' : 'empty',
    collectedAt,
    successfulMethods,
    totalMethods: METHODS.length,
    traffic,
    queries,
    pages,
    crawl,
    feeds,
    errors,
  }
}
