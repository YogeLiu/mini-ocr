// 把课程的 Markdown 同步到 Starlight 内容目录。
// 源文件保持不变：补 frontmatter、去掉 H1（由标题栏渲染）、改写站内链接。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, '../..');
const out = path.resolve(here, '../src/content/docs');
const REPO = 'https://github.com/YogeLiu/mini-ocr/blob/main/course-v4/';
const BASE = process.env.SITE_BASE ?? '/mini-ocr';

const PARTS = [
  ['part1_foundations', '第一部分：基础'],
  ['part2_lm', '第二部分：语言模型'],
  ['part3_post_training', '第三部分：后训练'],
  ['part4_vlm', '第四部分：视觉语言模型'],
];
const dirs = (rel) =>
  fs.readdirSync(path.join(src, rel), { withFileTypes: true }).filter((e) => e.isDirectory() && e.name !== '__pycache__').map((e) => e.name).sort();
const exists = (rel) => fs.existsSync(path.join(src, rel));

// 源文件（相对 course-v4）→ 站内 slug
const pages = new Map([['README.md', 'index']]);
const add = (rel, slug) => { if (exists(rel)) pages.set(rel, slug); };

function addChapter(rel, name) {
  add(`${rel}/chapter.md`, name);
  add(`${rel}/lab.md`, `${name}-lab`);
  add(`${rel}/exercises.md`, `${name}-exercises`);
}

addChapter('book/ch00_intro', 'ch00_intro');
for (const [part] of PARTS) {
  for (const ch of dirs(`book/${part}`).filter((d) => d.startsWith('ch'))) addChapter(`book/${part}/${ch}`, ch);
  add(`book/${part}/self_test.md`, `${part}-self-test`);
}
add('book/appendix/README.md', 'appendix');
for (const a of dirs('book/appendix')) addChapter(`book/appendix/${a}`, `appendix-${a}`);
add('assignments/README.md', 'assignments');
for (const a of dirs('assignments').filter((d) => /^a\d+$/.test(d))) add(`assignments/${a}/handout.md`, `assignments-${a}`);
add('project/README.md', 'project');

const broken = [];

function rewriteHref(file, href) {
  if (/^(https?:|#|mailto:)/.test(href)) return null;
  const [target, hash] = href.split('#');
  const rel = path.posix.normalize(path.posix.join(path.posix.dirname(file), target));
  const slug = pages.get(rel);
  const anchor = hash ? `#${hash}` : '';
  if (slug === 'index') return `${BASE}/${anchor}`;
  if (slug) return `${BASE}/${slug}/${anchor}`;
  if (rel.startsWith('outputs/')) return { plain: true };
  if (!exists(rel)) { broken.push(`${file} -> ${href}`); return { plain: true }; }
  return `${REPO}${rel}${anchor}`;
}

// 只改正文里的链接，跳过代码块和行内代码。
function rewriteLinks(file, text) {
  let fence = null;
  return text.split('\n').map((line) => {
    const m = line.match(/^\s*(`{3,}|~{3,})/);
    if (m) {
      if (!fence) fence = m[1];
      else if (m[1].startsWith(fence)) fence = null;
      return line;
    }
    if (fence) return line;
    return line.split(/(`[^`]*`)/).map((seg, i) => {
      if (i % 2) return seg;
      return seg.replace(/\[([^\]]*)\]\(([^)\s]+)\)/g, (all, label, href) => {
        const to = rewriteHref(file, href);
        if (to === null) return all;
        return typeof to === 'string' ? `[${label}](${to})` : label;
      });
    }).join('');
  }).join('\n');
}

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

const titles = new Map();
for (const [file, slug] of pages) {
  let text = fs.readFileSync(path.join(src, file), 'utf8');
  const m = text.match(/^#\s+(.+)\n/);
  const title = m ? m[1].trim() : slug;
  if (m) text = text.slice(m[0].length);
  titles.set(file, title);
  const fm = { title, ...(slug === 'index' ? { tableOfContents: false } : {}) };
  const head = Object.entries(fm).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join('\n');
  fs.writeFileSync(path.join(out, `${slug}.md`), `---\n${head}\n---\n\n${rewriteLinks(file, text)}`);
}

function chapterGroup(rel) {
  const items = [['chapter.md', '正文'], ['lab.md', '实验'], ['exercises.md', '习题']]
    .filter(([f]) => pages.has(`${rel}/${f}`))
    .map(([f, label]) => ({ slug: pages.get(`${rel}/${f}`), label }));
  const label = titles.get(`${rel}/chapter.md`).replace(/^第\s*(\d+)\s*章\s*/, '$1 ').replace(/^附录\s*([A-Z])\s*/, '$1 ');
  return items.length === 1 ? { slug: items[0].slug, label } : { label, collapsed: true, items };
}

const sidebar = [{ label: '导论', collapsed: true, items: [chapterGroup('book/ch00_intro')] }];
for (const [part, label] of PARTS) {
  const items = dirs(`book/${part}`).filter((d) => d.startsWith('ch')).map((ch) => chapterGroup(`book/${part}/${ch}`));
  if (pages.has(`book/${part}/self_test.md`)) items.push({ slug: `${part}-self-test`, label: '闭卷自测' });
  sidebar.push({ label, collapsed: true, items });
}
sidebar.push({
  label: '大作业与项目',
  collapsed: true,
  items: [
    { slug: 'assignments', label: '作业说明' },
    ...[...pages].filter(([, s]) => /^assignments-a\d+$/.test(s)).map(([, s]) => ({ slug: s, label: s.replace('assignments-a', 'A') })),
    { slug: 'project', label: 'TexPixel 项目' },
  ],
});
sidebar.push({
  label: '附录',
  collapsed: true,
  items: [{ slug: 'appendix', label: '说明' }, ...dirs('book/appendix').map((a) => chapterGroup(`book/appendix/${a}`))],
});
fs.writeFileSync(path.resolve(here, '../sidebar.json'), JSON.stringify(sidebar, null, 2));

console.log(`synced ${pages.size} pages`);
if (broken.length) console.warn(`broken links (rendered as plain text):\n  ${broken.join('\n  ')}`);
