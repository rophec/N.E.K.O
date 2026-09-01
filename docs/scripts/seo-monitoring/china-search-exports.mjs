import { readdir, readFile, stat } from 'node:fs/promises'
import { basename, extname, join, resolve } from 'node:path'

const SUPPORTED_EXTENSIONS = new Set(['.csv', '.tsv', '.txt', '.json'])

const ALIASES = {
  date: ['日期', '时间', 'date', 'day'],
  keyword: ['关键词', '搜索词', '查询词', 'query', 'keyword'],
  page: ['页面', '网页', '落地页', '链接', 'url', 'page'],
  clicks: ['点击量', '点击次数', '点击', '访问次数', 'clicks', 'click'],
  impressions: ['展现量', '展示量', '曝光量', '搜索展现量', 'impressions', 'impression'],
  ctr: ['点击率', 'ctr'],
  position: ['平均排名', '排名', '平均位置', 'position', 'rank'],
  indexed: ['索引量', '收录量', '收录', 'indexed', 'index'],
}

function normalizedHeader(value) {
  return String(value ?? '').replace(/^\uFEFF/u, '').trim().toLocaleLowerCase('zh-CN')
}

function findHeader(headers, aliases) {
  const wanted = new Set(aliases.map(normalizedHeader))
  return headers.find(header => wanted.has(normalizedHeader(header))) ?? null
}

function numeric(value, { percentage = false } = {}) {
  const text = String(value ?? '').trim().replaceAll(',', '')
  if (!text || ['-', '--', 'n/a'].includes(text.toLocaleLowerCase('en-US'))) return null
  const number = Number.parseFloat(text.replace('%', ''))
  if (!Number.isFinite(number)) return null
  return percentage && text.includes('%') ? number / 100 : number
}

function isoDate(value) {
  const text = String(value ?? '').trim()
  const match = text.match(/^(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})/u)
  if (match) return `${match[1]}-${match[2].padStart(2, '0')}-${match[3].padStart(2, '0')}`
  const date = new Date(text)
  return Number.isNaN(date.getTime()) ? null : date.toISOString().slice(0, 10)
}

function parseDelimitedLine(line, delimiter) {
  const values = []
  let value = ''
  let quoted = false
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index]
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"'
        index += 1
      } else quoted = !quoted
    } else if (character === delimiter && !quoted) {
      values.push(value.trim())
      value = ''
    } else value += character
  }
  values.push(value.trim())
  return values
}

export function parseDelimitedExport(text) {
  const lines = String(text).replace(/^\uFEFF/u, '').split(/\r?\n/u).filter(line => line.trim())
  if (lines.length === 0) return []
  const delimiter = (lines[0].match(/\t/gu)?.length ?? 0) > (lines[0].match(/,/gu)?.length ?? 0) ? '\t' : ','
  const headers = parseDelimitedLine(lines[0], delimiter)
  return lines.slice(1).map(line => {
    const values = parseDelimitedLine(line, delimiter)
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']))
  })
}

function normalizeRows(rows) {
  const headers = [...new Set(rows.flatMap(row => Object.keys(row)))]
  const columns = Object.fromEntries(Object.entries(ALIASES).map(([name, aliases]) => [name, findHeader(headers, aliases)]))
  return rows.map(row => ({
    date: columns.date ? isoDate(row[columns.date]) : null,
    keyword: columns.keyword ? String(row[columns.keyword] ?? '').trim() || null : null,
    page: columns.page ? String(row[columns.page] ?? '').trim() || null : null,
    clicks: columns.clicks ? numeric(row[columns.clicks]) : null,
    impressions: columns.impressions ? numeric(row[columns.impressions]) : null,
    ctr: columns.ctr ? numeric(row[columns.ctr], { percentage: true }) : null,
    position: columns.position ? numeric(row[columns.position]) : null,
    indexed: columns.indexed ? numeric(row[columns.indexed]) : null,
  }))
}

function addDays(date, days) {
  const result = new Date(`${date}T00:00:00.000Z`)
  result.setUTCDate(result.getUTCDate() + days)
  return result.toISOString().slice(0, 10)
}

function sumTraffic(rows) {
  return rows.reduce((total, row) => ({
    clicks: total.clicks + (row.clicks ?? 0),
    impressions: total.impressions + (row.impressions ?? 0),
  }), { clicks: 0, impressions: 0 })
}

function trafficSummary(rows, fallbackDate) {
  const daily = rows.filter(row => row.date && (row.clicks != null || row.impressions != null))
    .sort((left, right) => left.date.localeCompare(right.date))
  const dataThrough = daily.at(-1)?.date ?? fallbackDate
  if (daily.length === 0) return { dataThrough, recent7: null, previous7: null, trend7: null, daily: [] }
  const recentStart = addDays(dataThrough, -6)
  const previousEnd = addDays(recentStart, -1)
  const previousStart = addDays(previousEnd, -6)
  const recent7 = sumTraffic(daily.filter(row => row.date >= recentStart && row.date <= dataThrough))
  const previous7 = sumTraffic(daily.filter(row => row.date >= previousStart && row.date <= previousEnd))
  return {
    dataThrough,
    recent7,
    previous7,
    trend7: {
      clicks: recent7.clicks - previous7.clicks,
      impressions: recent7.impressions - previous7.impressions,
    },
    daily: daily.slice(-35),
  }
}

function topItems(rows, field) {
  return rows.filter(row => row[field]).sort((left, right) =>
    (right.clicks ?? 0) - (left.clicks ?? 0)
    || (right.impressions ?? 0) - (left.impressions ?? 0)
    || String(left[field]).localeCompare(String(right[field]), 'zh-CN'))
    .slice(0, 25)
    .map(row => ({
      [field]: row[field],
      ...(field === 'page' ? {} : { page: row.page }),
      clicks: row.clicks,
      impressions: row.impressions,
      ctr: row.ctr ?? (row.clicks != null && row.impressions ? row.clicks / row.impressions : null),
      position: row.position,
    }))
}

async function latestExport(path) {
  const absolute = resolve(path)
  const info = await stat(absolute)
  if (info.isFile()) return { path: absolute, info }
  const candidates = await Promise.all((await readdir(absolute, { withFileTypes: true }))
    .filter(entry => entry.isFile() && SUPPORTED_EXTENSIONS.has(extname(entry.name).toLocaleLowerCase('en-US')))
    .map(async entry => {
      const candidatePath = join(absolute, entry.name)
      return { path: candidatePath, info: await stat(candidatePath) }
    }))
  candidates.sort((left, right) => right.info.mtimeMs - left.info.mtimeMs)
  return candidates[0] ?? null
}

async function readRows(path) {
  const text = await readFile(path, 'utf8')
  if (extname(path).toLocaleLowerCase('en-US') === '.json') {
    const parsed = JSON.parse(text)
    if (Array.isArray(parsed)) return parsed
    if (Array.isArray(parsed.rows)) return parsed.rows
    throw new TypeError('JSON export must be an array or contain a rows array')
  }
  return parseDelimitedExport(text)
}

export async function collectChinaSearchExport(definition, { exportPath } = {}) {
  if (!exportPath) throw new TypeError(`${definition.platform} export path is required`)
  const file = await latestExport(exportPath)
  if (!file) return {
    ...definition,
    status: 'ok',
    availability: 'empty',
    collectedAt: null,
    sourceFile: null,
    traffic: { dataThrough: null, recent7: null, previous7: null, trend7: null, daily: [] },
    keywords: [],
    pages: [],
    latestIndexed: null,
  }
  const rows = normalizeRows(await readRows(file.path))
  const collectedAt = file.info.mtime.toISOString()
  const fallbackDate = collectedAt.slice(0, 10)
  const traffic = trafficSummary(rows, fallbackDate)
  const keywords = topItems(rows, 'keyword')
  const pages = topItems(rows, 'page')
  const indexedRows = rows.filter(row => row.indexed != null).sort((left, right) => (left.date ?? '').localeCompare(right.date ?? ''))
  const hasData = Boolean(traffic.daily.length || keywords.length || pages.length || indexedRows.length)
  return {
    ...definition,
    status: 'ok',
    availability: hasData ? 'available' : 'empty',
    collectedAt,
    sourceFile: basename(file.path),
    traffic,
    keywords,
    pages,
    latestIndexed: indexedRows.at(-1)?.indexed ?? null,
  }
}
