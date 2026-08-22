const frontmatterFieldPattern = /^# ([a-z_]+):\s*(.*)$/;
const dependencyPattern = /^[A-Za-z0-9_.-]+$/;

/**
 * @typedef {object} TutorialMetadata
 * @property {number} order
 * @property {string} title
 * @property {string[]} deps
 * @property {string} content
 */

/**
 * @param {string} source
 * @param {string} tutorialPath
 * @returns {TutorialMetadata}
 */
export function parseTutorialMetadata(source, tutorialPath) {
  const lines = source.split(/\r?\n/);
  if (lines[0] !== '# ---') {
    throw new Error(`${tutorialPath} must start with tutorial frontmatter`);
  }
  const frontmatterEnd = lines.indexOf('# ---', 1);
  if (frontmatterEnd === -1) {
    throw new Error(`${tutorialPath} has unterminated tutorial frontmatter`);
  }

  const fields = new Map();
  for (const line of lines.slice(1, frontmatterEnd)) {
    const match = line.match(frontmatterFieldPattern);
    if (!match) {
      throw new Error(`${tutorialPath} has invalid frontmatter line: ${line}`);
    }
    const [, name, value] = match;
    if (name !== 'order' && name !== 'deps') {
      throw new Error(`${tutorialPath} has unsupported frontmatter field: ${name}`);
    }
    if (fields.has(name)) {
      throw new Error(`${tutorialPath} has duplicate frontmatter field: ${name}`);
    }
    fields.set(name, value);
  }

  const orderText = fields.get('order');
  if (!orderText || !/^\d+$/.test(orderText)) {
    throw new Error(`${tutorialPath} frontmatter requires a non-negative integer order`);
  }
  const order = Number(orderText);
  if (!Number.isSafeInteger(order)) {
    throw new Error(`${tutorialPath} frontmatter order exceeds the safe integer range`);
  }

  const deps = (fields.get('deps') ?? '')
    .split(',')
    .map((dependency) => dependency.trim())
    .filter(Boolean);
  if (new Set(deps).size !== deps.length) {
    throw new Error(`${tutorialPath} frontmatter deps must be unique`);
  }
  const invalidDeps = deps.filter((dependency) => !dependencyPattern.test(dependency));
  if (invalidDeps.length > 0) {
    throw new Error(`${tutorialPath} has invalid frontmatter deps: ${invalidDeps.join(', ')}`);
  }

  const contentLines = lines.slice(frontmatterEnd + 1);
  const titleLine = contentLines.find((line) => line.startsWith('# # '));
  if (!titleLine) {
    throw new Error(`${tutorialPath} is missing an H1 heading`);
  }

  return {
    order,
    title: titleLine.slice(4).trim(),
    deps,
    content: contentLines.join('\n'),
  };
}
