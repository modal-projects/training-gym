import { visit } from 'unist-util-visit';

export function rehypeLazyMedia() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img') {
        node.properties = { loading: 'lazy', decoding: 'async', ...node.properties };
      }
      if (node.tagName === 'video' && node.properties?.preload === undefined) {
        node.properties = { ...node.properties, preload: 'none' };
      }
    });
  };
}
