import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import { collectChinaSearchExport, parseDelimitedExport } from './china-search-exports.mjs'

const definition = {
  id: 'baidu-cn',
  platform: 'baidu',
  label: '.cn 产品主页',
  siteUrl: 'https://project-neko.cn/',
}

test('parser accepts BOM, quoted commas, and Chinese export headers', () => {
  const rows = parseDelimitedExport('\uFEFF关键词,点击量,展现量,点击率,平均排名,页面\n"AI,桌宠",2,"1,000",0.2%,8.5,https://project-neko.cn/')
  assert.deepEqual(rows, [{
    关键词: 'AI,桌宠',
    点击量: '2',
    展现量: '1,000',
    点击率: '0.2%',
    平均排名: '8.5',
    页面: 'https://project-neko.cn/',
  }])
})

test('collector reports an empty import directory as waiting for an export', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'neko-china-empty-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const result = await collectChinaSearchExport(definition, { exportPath: directory })

  assert.equal(result.status, 'not_run')
  assert.equal(result.availability, 'not_run')
  assert.match(result.reason, /等待本地导出/u)
})

test('collector reads the newest local export and computes consecutive seven-day windows', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'neko-china-search-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const lines = ['日期,点击量,展现量,索引量']
  for (let day = 1; day <= 14; day += 1) {
    lines.push(`2026-08-${String(day).padStart(2, '0')},${day <= 7 ? 1 : 2},${day <= 7 ? 10 : 20},${90 + day}`)
  }
  await writeFile(join(directory, 'baidu-traffic.csv'), `${lines.join('\n')}\n`, 'utf8')
  const result = await collectChinaSearchExport(definition, { exportPath: directory })

  assert.equal(result.status, 'ok')
  assert.equal(result.availability, 'available')
  assert.equal(result.sourceFile, 'baidu-traffic.csv')
  assert.equal(result.traffic.dataThrough, '2026-08-14')
  assert.deepEqual(result.traffic.recent7, { clicks: 14, impressions: 140 })
  assert.deepEqual(result.traffic.previous7, { clicks: 7, impressions: 70 })
  assert.deepEqual(result.traffic.trend7, { clicks: 7, impressions: 70 })
  assert.equal(result.latestIndexed, 104)
})

test('collector normalizes tab-separated keyword and page metrics', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'neko-china-keywords-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  await writeFile(join(directory, '360-keywords.tsv'), [
    '关键词\t页面\t点击次数\t展示量\t点击率\t排名',
    'AI桌宠\thttps://project-neko.cn/\t3\t30\t10%\t4',
    '桌面伴侣\thttps://project-neko.cn/\t1\t20\t5%\t9',
  ].join('\n'), 'utf8')

  const result = await collectChinaSearchExport({ ...definition, id: '360-cn', platform: '360' }, { exportPath: directory })
  assert.equal(result.keywords[0].keyword, 'AI桌宠')
  assert.equal(result.keywords[0].ctr, 0.1)
  assert.equal(result.keywords[0].position, 4)
  assert.equal(result.pages[0].page, 'https://project-neko.cn/')
})
