const NODE_W = 160;
const NODE_H = 44;
const GAP_X = 40;
const GAP_Y = 28;
const PAD = 60;

let modelData = null;
let selectedNode = null;
let activeResize = null; // tracks which panel is being resized

async function init() {
  const res = await fetch('/api/model');
  modelData = await res.json();

  document.getElementById('filename').textContent = modelData.fileName;
  document.getElementById('stats').textContent =
    `${modelData.graph.nodes.length} nodes | ${modelData.graph.edges.length} edges | ${modelData.tensors.length} tensors`;

  const positions = layoutGraph(modelData.graph);
  renderGraph(modelData.graph, positions);
  setupPanZoom();
  setupIOPanel();
}

/**
 * Simple layered layout algorithm.
 * Assigns each node to a layer based on longest path from inputs,
 * then spaces nodes within each layer.
 */
function layoutGraph(graph) {
  const { nodes, edges } = graph;
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const inEdges = new Map();
  const outEdges = new Map();

  for (const n of nodes) {
    inEdges.set(n.id, []);
    outEdges.set(n.id, []);
  }

  for (const e of edges) {
    if (inEdges.has(e.to) && outEdges.has(e.from)) {
      inEdges.get(e.to).push(e.from);
      outEdges.get(e.from).push(e.to);
    }
  }

  // find roots (no incoming edges)
  const roots = nodes.filter(n => inEdges.get(n.id).length === 0);

  // assign layers via BFS longest path
  const layer = new Map();
  const queue = roots.map(n => n.id);
  for (const id of queue) layer.set(id, 0);

  const visited = new Set();
  while (queue.length > 0) {
    const id = queue.shift();
    if (visited.has(id)) continue;
    visited.add(id);

    const currentLayer = layer.get(id);
    for (const next of outEdges.get(id)) {
      const existing = layer.get(next) ?? -1;
      layer.set(next, Math.max(existing, currentLayer + 1));
      queue.push(next);
    }
  }

  // nodes not reached by BFS (disconnected)
  for (const n of nodes) {
    if (!layer.has(n.id)) layer.set(n.id, 0);
  }

  // group by layer
  const layers = new Map();
  for (const [id, l] of layer) {
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l).push(id);
  }

  // compute positions
  const positions = new Map();
  const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
  const maxNodesInLayer = Math.max(...[...layers.values()].map(l => l.length));

  for (const l of sortedLayers) {
    const ids = layers.get(l);
    const layerWidth = ids.length * (NODE_W + GAP_X) - GAP_X;
    const totalWidth = maxNodesInLayer * (NODE_W + GAP_X) - GAP_X;
    const offsetX = (totalWidth - layerWidth) / 2;

    ids.forEach((id, i) => {
      positions.set(id, {
        x: PAD + offsetX + i * (NODE_W + GAP_X),
        y: PAD + l * (NODE_H + GAP_Y),
      });
    });
  }

  return positions;
}

function renderGraph(graph, positions) {
  const svg = document.getElementById('graph');
  const { nodes, edges } = graph;

  // compute SVG size
  let maxX = 0, maxY = 0;
  for (const pos of positions.values()) {
    maxX = Math.max(maxX, pos.x + NODE_W + PAD);
    maxY = Math.max(maxY, pos.y + NODE_H + PAD);
  }
  svg.setAttribute('width', maxX);
  svg.setAttribute('height', maxY);
  svg.setAttribute('viewBox', `0 0 ${maxX} ${maxY}`);

  // defs
  const defs = el('defs');
  defs.innerHTML = `
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#484f58"/>
    </marker>
  `;
  svg.appendChild(defs);

  // edges
  const edgeGroup = el('g', { class: 'edges' });
  for (const edge of edges) {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) continue;

    const x1 = from.x + NODE_W / 2;
    const y1 = from.y + NODE_H;
    const x2 = to.x + NODE_W / 2;
    const y2 = to.y;

    const midY = (y1 + y2) / 2;
    const path = el('path', {
      class: 'edge',
      d: `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`,
    });
    edgeGroup.appendChild(path);
  }
  svg.appendChild(edgeGroup);

  // nodes
  const nodeGroup = el('g', { class: 'nodes' });
  for (const node of nodes) {
    const pos = positions.get(node.id);
    if (!pos) continue;

    const typeClass = `type-${node.type || 'Default'}`;
    const g = el('g', {
      class: `node ${typeClass}`,
      transform: `translate(${pos.x}, ${pos.y})`,
      'data-id': node.id,
    });

    g.appendChild(el('rect', { width: NODE_W, height: NODE_H }));

    const shortName = shortenName(node.name);
    const label = el('text', {
      class: 'node-label',
      x: NODE_W / 2,
      y: 18,
    });
    label.textContent = shortName;
    g.appendChild(label);

    const typeLabel = el('text', {
      class: 'node-type',
      x: NODE_W / 2,
      y: 34,
    });
    typeLabel.textContent = node.type;
    g.appendChild(typeLabel);

    g.addEventListener('click', () => selectNode(node));
    nodeGroup.appendChild(g);
  }
  svg.appendChild(nodeGroup);
}

function shortenName(name) {
  if (name.length <= 20) return name;
  const parts = name.split('.');
  if (parts.length <= 2) return name.slice(0, 18) + '..';
  // keep first and last two parts
  return parts[0] + '..' + parts.slice(-2).join('.');
}

function selectNode(node) {
  // deselect previous
  document.querySelectorAll('.node.selected').forEach(n => n.classList.remove('selected'));

  const el = document.querySelector(`.node[data-id="${node.id}"]`);
  if (el) el.classList.add('selected');

  selectedNode = node;
  showDetails(node);
}

function showDetails(node) {
  const sidebar = document.getElementById('sidebar');
  const details = document.getElementById('node-details');
  sidebar.classList.remove('hidden');

  let html = `<h3>${node.name}</h3>`;
  html += `<div class="detail-type">${node.type}</div>`;

  if (node.params && node.params.length > 0) {
    html += `<div class="section-title">Parameters</div>`;
    html += `<table><tr><th>Name</th><th>Shape</th><th>Type</th></tr>`;
    for (const p of node.params) {
      const shape = p.shape ? `[${p.shape.join(', ')}]` : '-';
      html += `<tr><td>${p.name}</td><td>${shape}</td><td>${p.dtype || '-'}</td></tr>`;
    }
    html += `</table>`;
  }

  if (node.attributes && node.attributes.length > 0) {
    html += `<div class="section-title">Attributes</div>`;
    html += `<table><tr><th>Name</th><th>Value</th></tr>`;
    for (const a of node.attributes) {
      let val = '-';
      if (a.value != null) {
        if (Array.isArray(a.value)) {
          val = a.value.length > 10
            ? `[${a.value.slice(0, 10).join(', ')}, ... ${a.value.length} items]`
            : `[${a.value.join(', ')}]`;
        } else {
          val = String(a.value);
          if (val.length > 200) val = val.slice(0, 200) + '...';
        }
      }
      html += `<tr><td>${a.name}</td><td>${val}</td></tr>`;
    }
    html += `</table>`;
  }

  if (node.inputs && node.inputs.length > 0) {
    html += `<div class="section-title">Inputs</div>`;
    html += `<table><tr><th>Tensor</th></tr>`;
    for (const inp of node.inputs) {
      html += `<tr><td>${inp}</td></tr>`;
    }
    html += `</table>`;
  }

  if (node.outputs && node.outputs.length > 0) {
    html += `<div class="section-title">Outputs</div>`;
    html += `<table><tr><th>Tensor</th></tr>`;
    for (const out of node.outputs) {
      html += `<tr><td>${out}</td></tr>`;
    }
    html += `</table>`;
  }

  details.innerHTML = html;
}

function setupPanZoom() {
  const container = document.getElementById('canvas-container');
  const svg = document.getElementById('graph');
  let scale = 1;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let startX, startY;

  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.min(Math.max(scale * delta, 0.1), 5);
    svg.style.transform = `scale(${scale}) translate(${panX}px, ${panY}px)`;
    svg.style.transformOrigin = '0 0';
  }, { passive: false });

  container.addEventListener('mousedown', (e) => {
    if (e.target.closest('.node')) return;
    isPanning = true;
    startX = e.clientX - panX * scale;
    startY = e.clientY - panY * scale;
  });

  window.addEventListener('mousemove', (e) => {
    if (activeResize) return; // don't pan while resizing
    if (!isPanning) return;
    panX = (e.clientX - startX) / scale;
    panY = (e.clientY - startY) / scale;
    svg.style.transform = `scale(${scale}) translate(${panX}px, ${panY}px)`;
    svg.style.transformOrigin = '0 0';
  });

  window.addEventListener('mouseup', () => { isPanning = false; });

  document.getElementById('close-sidebar').addEventListener('click', () => {
    document.getElementById('sidebar').classList.add('hidden');
    document.querySelectorAll('.node.selected').forEach(n => n.classList.remove('selected'));
    selectedNode = null;
  });

  // sidebar resize
  const sidebar = document.getElementById('sidebar');
  const saved = localStorage.getItem('mlpeel-sidebar-width');
  if (saved) sidebar.style.width = saved + 'px';

  setupResizeHandle(
    document.getElementById('sidebar-handle'),
    sidebar,
    'mlpeel-sidebar-width',
    200,
  );
}

function setupIOPanel() {
  const panel = document.getElementById('io-panel');
  const toggleBtn = document.getElementById('toggle-io');
  const closeBtn = document.getElementById('close-io');
  const ioHandle = document.getElementById('io-handle');

  // find Input/Output nodes from graph
  const inputNodes = modelData.graph.nodes.filter(n => n.type === 'Input');
  const outputNodes = modelData.graph.nodes.filter(n => n.type === 'Output');

  // render input format table
  const inputsDiv = document.getElementById('io-inputs');
  if (inputNodes.length > 0) {
    let html = '<table class="io-table"><tr><th>Name</th><th>Shape</th><th>Type</th></tr>';
    for (const n of inputNodes) {
      const p = n.params?.[0] || {};
      const shape = p.shape ? `[${p.shape.join(', ')}]` : '-';
      html += `<tr><td>${n.name}</td><td class="mono">${shape}</td><td class="mono">${p.dtype || '-'}</td></tr>`;
    }
    html += '</table>';
    inputsDiv.innerHTML = html;
  } else {
    inputsDiv.innerHTML = '<p class="hint">No input info available</p>';
  }

  // render output format table
  const outputsDiv = document.getElementById('io-outputs');
  if (outputNodes.length > 0) {
    let html = '<table class="io-table"><tr><th>Name</th><th>Shape</th><th>Type</th></tr>';
    for (const n of outputNodes) {
      const p = n.params?.[0] || {};
      const shape = p.shape ? `[${p.shape.join(', ')}]` : '-';
      html += `<tr><td>${n.name}</td><td class="mono">${shape}</td><td class="mono">${p.dtype || '-'}</td></tr>`;
    }
    html += '</table>';
    outputsDiv.innerHTML = html;
  } else {
    outputsDiv.innerHTML = '<p class="hint">No output info available</p>';
  }

  // inference section (ONNX only)
  if (modelData.canInfer) {
    document.getElementById('infer-section').classList.remove('hidden');

    // pre-fill input template
    const template = {};
    for (const n of inputNodes) {
      const p = n.params?.[0] || {};
      const shape = p.shape || [];
      const dtype = (p.dtype || 'FLOAT').toLowerCase();
      const type = dtype.includes('float') ? 'float32'
        : dtype.includes('int64') ? 'int64'
        : dtype.includes('int') ? 'int32'
        : dtype.includes('double') ? 'float64'
        : 'float32';
      const size = shape.reduce((a, b) => (typeof b === 'number' ? a * b : a), 1);
      template[n.name] = {
        data: Array(Math.min(size, 20)).fill(0),
        shape: shape.map(d => typeof d === 'number' ? d : 1),
        type,
      };
    }
    document.getElementById('infer-input').value = JSON.stringify(template, null, 2);

    // run button
    document.getElementById('infer-btn').addEventListener('click', async () => {
      const btn = document.getElementById('infer-btn');
      const status = document.getElementById('infer-status');
      const output = document.getElementById('infer-output');
      const resultTitle = document.querySelector('.result-title');

      btn.disabled = true;
      status.className = '';
      status.textContent = 'Running...';
      output.textContent = '';
      resultTitle.classList.add('hidden');

      try {
        const inputJson = JSON.parse(document.getElementById('infer-input').value);
        const res = await fetch('/api/infer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputJson),
        });
        const data = await res.json();

        if (data.error) {
          status.className = 'error';
          status.textContent = data.error;
        } else {
          status.className = 'ok';
          status.textContent = 'Done';
          resultTitle.classList.remove('hidden');
          // format output: truncate large arrays
          const formatted = {};
          for (const [k, v] of Object.entries(data)) {
            formatted[k] = {
              shape: v.shape,
              type: v.type,
              data: v.data.length > 50
                ? [...v.data.slice(0, 50), `... ${v.data.length} total`]
                : v.data,
            };
          }
          output.textContent = JSON.stringify(formatted, null, 2);
        }
      } catch (err) {
        status.className = 'error';
        status.textContent = err.message;
      }
      btn.disabled = false;
    });
  }

  // toggle panel
  toggleBtn.addEventListener('click', () => {
    const isHidden = panel.classList.toggle('hidden');
    toggleBtn.classList.toggle('active', !isHidden);
  });

  closeBtn.addEventListener('click', () => {
    panel.classList.add('hidden');
    toggleBtn.classList.remove('active');
  });

  // resize handle
  const savedIO = localStorage.getItem('mlpeel-io-width');
  if (savedIO) panel.style.width = savedIO + 'px';

  setupResizeHandle(ioHandle, panel, 'mlpeel-io-width', 280);
}

function setupResizeHandle(handle, panel, storageKey, minWidth) {
  handle.addEventListener('mousedown', (e) => {
    activeResize = { handle, panel, storageKey, minWidth };
    handle.classList.add('dragging');
    e.preventDefault();
  });
}

window.addEventListener('mousemove', (e) => {
  if (!activeResize) return;
  const right = activeResize.panel.getBoundingClientRect().right;
  const w = Math.max(activeResize.minWidth, Math.min(right - e.clientX, window.innerWidth * 0.6));
  activeResize.panel.style.width = w + 'px';
});

window.addEventListener('mouseup', () => {
  if (!activeResize) return;
  activeResize.handle.classList.remove('dragging');
  localStorage.setItem(activeResize.storageKey, parseInt(activeResize.panel.style.width));
  activeResize = null;
});

function el(tag, attrs = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) element.setAttribute(k, v);
  return element;
}

init();
