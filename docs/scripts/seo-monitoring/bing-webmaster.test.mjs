import assert from 'node:assert/strict'
import test from 'node:test'

import { collectBingWebmasterSite, parseBingDate } from './bing-webmaster.mjs'

const definition = {
  id: 'online',
  label: '.online 文档站',
  siteUrl: 'https://project-neko.online/',
}

function dotNetDate(date) {
  return `/Date(${Date.parse(`${date}T00:00:00.000Z`)})/`
}

function response(rows, status = 200) {
  return new Response(JSON.stringify({ d: rows }), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function methodFromUrl(url) {
  return new URL(url).pathname.split('/').at(-1)
}

function verifiedSite(url = definition.siteUrl) {
  return [{ Url: url, IsVerified: true }]
}

test('parseBingDate parses .NET JSON dates', () => {
  assert.equal(parseBingDate('/Date(1787616000000+0000)/'), '2026-08-25')
  assert.equal(parseBingDate('invalid'), null)
})

test('collector computes daily windows and uses only the latest weekly snapshot', async () => {
  const traffic = Array.from({ length: 14 }, (_, index) => ({
    Date: dotNetDate(`2026-08-${String(index + 1).padStart(2, '0')}`),
    Clicks: index < 7 ? 1 : 2,
    Impressions: index < 7 ? 10 : 20,
  }))
  const payloads = {
    GetUserSites: verifiedSite(),
    GetRankAndTrafficStats: traffic,
    GetQueryStats: [
      { Date: dotNetDate('2026-08-07'), Query: 'old query', Clicks: 99, Impressions: 999 },
      { Date: dotNetDate('2026-08-14'), Query: 'second query', Clicks: 2, Impressions: 30, AvgClickPosition: 3, AvgImpressionPosition: 4 },
      { Date: dotNetDate('2026-08-14'), Query: 'top query', Clicks: 5, Impressions: 20, AvgClickPosition: 1, AvgImpressionPosition: 2 },
    ],
    GetPageStats: [
      { Date: dotNetDate('2026-08-14'), Query: 'https://project-neko.online/', Clicks: 3, Impressions: 40 },
    ],
    GetCrawlStats: [
      { Date: dotNetDate('2026-08-14'), CrawledPages: 8, InIndex: 100, CrawlErrors: 1, Code2xx: 7, Code4xx: 1 },
    ],
    GetFeeds: [
      { Url: 'https://project-neko.online/sitemap.xml', Status: 'Success', UrlCount: 308, LastCrawled: dotNetDate('2026-08-13') },
    ],
  }
  const result = await collectBingWebmasterSite(definition, {
    apiKey: 'secret-key',
    fetchImpl: async url => response(payloads[methodFromUrl(url)]),
  })

  assert.equal(result.status, 'ok')
  assert.equal(result.availability, 'available')
  assert.deepEqual(result.traffic.recent7, { clicks: 14, impressions: 140 })
  assert.deepEqual(result.traffic.previous7, { clicks: 7, impressions: 70 })
  assert.deepEqual(result.traffic.trend7, { clicks: 7, impressions: 70 })
  assert.equal(result.queries.dataThrough, '2026-08-14')
  assert.deepEqual(result.queries.items.map(item => item.query), ['top query', 'second query'])
  assert.equal(result.pages.items[0].page, 'https://project-neko.online/')
  assert.equal(result.crawl.latestDay.inIndex, 100)
  assert.equal(result.feeds[0].urlCount, 308)
})

test('verified site with successful empty responses is not treated as a failure', async () => {
  const result = await collectBingWebmasterSite(definition, {
    apiKey: 'secret-key',
    fetchImpl: async url => response(methodFromUrl(url) === 'GetUserSites' ? verifiedSite() : []),
  })
  assert.equal(result.status, 'ok')
  assert.equal(result.availability, 'empty')
  assert.equal(result.isVerified, true)
  assert.equal(result.errors.length, 0)
})

test('site listed without ownership verification does not collect statistics', async () => {
  const methods = []
  const result = await collectBingWebmasterSite(definition, {
    apiKey: 'secret-key',
    fetchImpl: async url => {
      methods.push(methodFromUrl(url))
      return response([{ Url: definition.siteUrl, IsVerified: false }])
    },
  })
  assert.equal(result.status, 'not_verified')
  assert.equal(result.availability, 'not_verified')
  assert.equal(result.isVerified, false)
  assert.deepEqual(methods, ['GetUserSites'])
})

test('one endpoint failure keeps partial data and redacts the API key', async () => {
  const apiKey = 'never-print-this-key'
  const result = await collectBingWebmasterSite(definition, {
    apiKey,
    fetchImpl: async url => {
      if (methodFromUrl(url) === 'GetUserSites') return response(verifiedSite())
      if (methodFromUrl(url) === 'GetQueryStats') throw new Error(`network failed for ${apiKey}`)
      return response(methodFromUrl(url) === 'GetRankAndTrafficStats'
        ? [{ Date: dotNetDate('2026-08-14'), Clicks: 1, Impressions: 9 }]
        : [])
    },
  })
  assert.equal(result.status, 'partial')
  assert.equal(result.traffic.latestDay.clicks, 1)
  assert.equal(result.errors.length, 1)
  assert.equal(JSON.stringify(result).includes(apiKey), false)
  assert.match(result.errors[0].reason, /\[REDACTED\]/u)
})
