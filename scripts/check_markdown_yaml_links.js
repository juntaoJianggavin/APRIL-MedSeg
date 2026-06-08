#!/usr/bin/env node
/* Check Markdown links that point to local YAML files. */

const fs = require("fs");
const path = require("path");

const root = process.cwd();

function walk(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === ".git") continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
    } else if (/\.md$/i.test(entry.name)) {
      files.push(fullPath);
    }
  }
  return files;
}

const linkRe = /\[[^\]]*\]\(([^)\s]+\.ya?ml(?:#[^)\s]+)?)(?:\s+"[^"]*")?\)/gi;
const broken = [];
let checked = 0;

for (const mdFile of walk(root)) {
  const text = fs.readFileSync(mdFile, "utf8");
  let match;
  while ((match = linkRe.exec(text))) {
    const target = match[1].split("#")[0];
    if (/^(https?:|mailto:)/i.test(target)) continue;

    checked += 1;
    const resolved = path.resolve(path.dirname(mdFile), decodeURIComponent(target));
    if (!fs.existsSync(resolved)) {
      broken.push({
        file: path.relative(root, mdFile),
        line: text.slice(0, match.index).split("\n").length,
        target,
        resolved: path.relative(root, resolved),
      });
    }
  }
}

if (broken.length > 0) {
  console.error(`Broken Markdown YAML links: ${broken.length}/${checked}`);
  for (const item of broken) {
    console.error(`${item.file}:${item.line} -> ${item.target} (${item.resolved})`);
  }
  process.exit(1);
}

console.log(`Markdown YAML links OK: ${checked} checked.`);
