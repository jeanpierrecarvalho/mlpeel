/**
 * Safetensors parser.
 *
 * Format: [8 bytes header_len (LE uint64)] [JSON header] [tensor data]
 * The JSON header maps tensor names to { dtype, shape, data_offsets }.
 */

const DTYPE_SIZES = {
  F16: 2, BF16: 2, F32: 4, F64: 8,
  I8: 1, I16: 2, I32: 4, I64: 8,
  U8: 1, U16: 2, U32: 4, U64: 8,
  BOOL: 1,
};

/**
 * @param {Buffer} buffer
 * @returns {{ metadata: object, tensors: object[], graph: object }}
 */
export function parse(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));

  if (headerLen > buffer.length - 8) {
    throw new Error('Invalid safetensors file: header length exceeds file size');
  }

  const headerJson = buffer.subarray(8, 8 + Number(headerLen)).toString('utf-8');
  const header = JSON.parse(headerJson);

  const metadata = header.__metadata__ || {};
  delete header.__metadata__;

  const tensors = Object.entries(header).map(([name, info]) => {
    const elements = info.shape.reduce((a, b) => a * b, 1);
    const byteSize = elements * (DTYPE_SIZES[info.dtype] || 4);
    return {
      name,
      dtype: info.dtype,
      shape: info.shape,
      offsets: info.data_offsets,
      byteSize,
    };
  });

  // sort by offset for sequential layout
  tensors.sort((a, b) => (a.offsets?.[0] ?? 0) - (b.offsets?.[0] ?? 0));

  // build a simple graph: group tensors by layer prefix
  const graph = buildGraph(tensors);

  return { metadata, tensors, graph };
}

function buildGraph(tensors) {
  const layers = new Map();

  for (const t of tensors) {
    // typical naming: "model.layers.0.self_attn.q_proj.weight"
    const parts = t.name.split('.');
    // group by everything except last part (weight/bias)
    const layerKey = parts.length > 1 ? parts.slice(0, -1).join('.') : t.name;
    const param = parts.length > 1 ? parts[parts.length - 1] : 'data';

    if (!layers.has(layerKey)) {
      layers.set(layerKey, { name: layerKey, type: guessType(layerKey), params: [] });
    }
    layers.get(layerKey).params.push({
      name: param,
      dtype: t.dtype,
      shape: t.shape,
      byteSize: t.byteSize,
    });
  }

  const nodes = [...layers.values()].map((layer, i) => ({
    id: `n${i}`,
    name: layer.name,
    type: layer.type,
    params: layer.params,
  }));

  // sequential edges based on layer order
  const edges = [];
  for (let i = 1; i < nodes.length; i++) {
    edges.push({ from: nodes[i - 1].id, to: nodes[i].id });
  }

  return { nodes, edges };
}

function guessType(name) {
  const lower = name.toLowerCase();
  if (lower.includes('embed')) return 'Embedding';
  if (lower.includes('attn') || lower.includes('attention')) return 'Attention';
  if (lower.includes('mlp') || lower.includes('ffn') || lower.includes('dense')) return 'MLP';
  if (lower.includes('norm') || lower.includes('ln')) return 'Normalization';
  if (lower.includes('lm_head') || lower.includes('output')) return 'Output';
  if (lower.includes('conv')) return 'Convolution';
  if (lower.includes('pool')) return 'Pooling';
  return 'Linear';
}
