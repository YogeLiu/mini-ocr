// 把上级目录的 ch*.md / appendix_*.md / README.md 同步到 Starlight 内容目录。
// 源文件保持不变:补 frontmatter、去掉 H1(由标题栏渲染)、改写站内链接。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, '../..');
const out = path.resolve(here, '../src/content/docs');
const REPO = 'https://github.com/YogeLiu/mini-ocr/blob/main/course-v3/';
const BASE = process.env.SITE_BASE ?? '/mini-ocr';

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

const pages = new Set(
  fs.readdirSync(src).filter((f) => /^(ch\d+|appendix_[a-z]).*\.md$/.test(f)).map((f) => f.replace(/\.md$/, '')),
);

function rewriteLinks(text) {
  return text.replace(/\]\(([^)\s]+)\)/g, (m, href) => {
    if (/^(https?:|#|mailto:)/.test(href)) return m;
    const [file, hash] = href.split('#');
    const stem = file.replace(/\.md$/, '');
    if (file.endsWith('.md') && pages.has(stem)) return `](${BASE}/${stem}/${hash ? '#' + hash : ''})`;
    if (file.endsWith('.md') && stem === 'README') return `](${BASE}/)`;
    return `](${REPO}${file.replace(/^\.\//, '')})`;
  });
}

function convert(file, slug, extra = {}) {
  let text = fs.readFileSync(path.join(src, file), 'utf8');
  const m = text.match(/^#\s+(.+)\n/);
  const title = m ? m[1].trim() : slug;
  if (m) text = text.slice(m[0].length);
  text = rewriteLinks(text);
  const fm = { title, ...extra };
  const head = Object.entries(fm).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join('\n');
  fs.writeFileSync(path.join(out, slug + '.md'), `---\n${head}\n---\n\n${text}`);
  return title;
}

const titles = {};
for (const stem of [...pages].sort()) titles[stem] = convert(stem + '.md', stem);
convert('README.md', 'index', { description: '系统学习视觉语言模型与 OCR 的自学教程', tableOfContents: false });

const groups = [
  ['开始', (n) => n === 0],
  ['一、数据与数学基础', (n) => n >= 1 && n <= 6],
  ['二、训练神经网络', (n) => n >= 7 && n <= 11],
  ['三、视觉与序列', (n) => n >= 12 && n <= 13],
  ['四、语言模型与 Transformer', (n) => n >= 14 && n <= 18],
  ['五、视觉语言模型', (n) => n >= 19 && n <= 25],
  ['六、后训练', (n) => n >= 26 && n <= 32],
  ['七、OCR 工程与案例', (n) => n >= 33],
];
const chapters = [...pages].filter((p) => p.startsWith('ch')).sort();
const sidebar = groups.map(([label, f]) => ({
  label,
  collapsed: true,
  items: chapters.filter((p) => f(Number(p.match(/^ch(\d+)/)[1]))).map((p) => ({ slug: p, label: titles[p].replace(/^第\s*(\d+)\s*章\s*/, '$1 ') })),
}));
sidebar.push({
  label: '附录',
  collapsed: true,
  items: [...pages].filter((p) => p.startsWith('appendix')).sort().map((p) => ({ slug: p, label: titles[p].replace(/^附录\s*([A-Z])\s*/, '$1 ') })),
});
fs.writeFileSync(path.resolve(here, '../sidebar.json'), JSON.stringify(sidebar, null, 2));
console.log(`synced ${pages.size} pages`);
