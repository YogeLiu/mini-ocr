// 在构建时识别课程里固定的写法,标成 Obsidian / 飞书风格的彩色块。源 .md 不变。
const text = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(text).join(''));
const level = (n) => (n.type === 'element' && /^h[1-6]$/.test(n.tagName) ? Number(n.tagName[1]) : 0);
const el = (tagName, className, children) => ({ type: 'element', tagName, properties: { className }, children });
const addClass = (n, c) => { n.properties ??= {}; n.properties.className = [...(n.properties.className ?? []), ...[].concat(c)]; };

const SECTION_KIND = [
  [/^学习目标/, 'goal'],
  [/^前置检查/, 'check'],
  [/^(\d+\.\d+\s*)?本章小结/, 'summary'],
  [/^(自测题|参考答案)/, 'quiz'],
  [/^术语卡/, 'terms'],
];
const LABEL_KIND = [
  [/^(运行前先预测|运行之前先预测|预测|先预测)$/, 'predict'],
  [/^(做什么|做法|手算|时间|思路)$/, 'do'],
  [/^(看什么|跑完后看什么|结果)$/, 'look'],
  [/^(症状|原因|问题)$/, 'warn'],
];
const QUOTE_KIND = [
  [/^(注意|重要|常见错误|一个容易踩的坑)/, 'warn'],
  [/^(答案)/, 'tip'],
];

// 把 children[i] 这个标题连同后面直到同级或更高级标题之前的兄弟节点,包进一个容器
function wrapSections(children, match) {
  const out = [];
  for (let i = 0; i < children.length; i++) {
    const n = children[i];
    const lv = level(n);
    const kind = lv ? match(n, lv) : null;
    if (!kind) { out.push(n); continue; }
    const body = [n];
    while (i + 1 < children.length && !(level(children[i + 1]) && level(children[i + 1]) <= lv)) body.push(children[++i]);
    const tail = [];
    while (body.length > 1 && body[body.length - 1].tagName === 'hr') tail.unshift(body.pop());
    out.push(el('div', ['block', ...kind], body), ...tail);
  }
  return out;
}

export default function rehypeCallouts() {
  return (tree) => {
    // 章内特殊小节(h2)
    tree.children = wrapSections(tree.children, (n, lv) => {
      if (lv !== 2) return null;
      const hit = SECTION_KIND.find(([re]) => re.test(text(n).trim()));
      return hit ? ['section-card', `section-${hit[1]}`] : null;
    });
    // 练习(h3/h4),可能嵌在上面的卡片之外;只处理顶层
    tree.children = wrapSections(tree.children, (n, lv) => (lv >= 3 && /^练习/.test(text(n).trim()) ? ['exercise'] : null));

    const walk = (n) => {
      if (n.type !== 'element') return n.children?.forEach(walk);
      if (n.tagName === 'blockquote') {
        const t = text(n).trim().replace(/^\*+/, '');
        const hit = QUOTE_KIND.find(([re]) => re.test(t));
        addClass(n, ['callout', `callout-${hit ? hit[1] : 'note'}`]);
      }
      if (/^h[34]$/.test(n.tagName)) {
        const t = text(n).trim();
        if (/^必做/.test(t)) addClass(n, 'tag-must');
        else if (/^选做/.test(t)) addClass(n, 'tag-opt');
      }
      if (n.tagName === 'p') {
        const first = n.children?.[0];
        if (first?.type === 'element' && first.tagName === 'strong') {
          const label = text(first).trim();
          const hit = LABEL_KIND.find(([re]) => re.test(label));
          if (hit) addClass(first, ['label', `label-${hit[1]}`]);
        }
      }
      n.children?.forEach(walk);
    };
    walk(tree);
  };
}
