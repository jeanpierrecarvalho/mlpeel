/**
 * GGUF parser.
 *
 * Format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 * Header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
 * Then: metadata key-value pairs, then tensor info entries.
 */

const GGUF_MAGIC = 0x46554747; // "GGUF" in LE
const GGML_TYPES = [
  'F32', 'F16', 'Q4_0', 'Q4_1', null, 'Q5_0', 'Q5_1', 'Q8_0', 'Q8_1',
  'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K', 'IQ2_XXS', 'IQ2_XS',
  'IQ3_XXS', 'IQ1_S', 'IQ4_NL', 'IQ3_S', 'IQ2_S', 'IQ4_XS', 'I8', 'I16',
  'I32', 'I64', 'F64', 'IQ1_M', 'BF16',
];

const VALUE_TYPES = {
  0: 'uint8', 1: 'int8', 2: 'uint16', 3: 'int16', 4: 'uint32', 5: 'int32',
  6: 'float32', 7: 'bool', 8: 'string', 9: 'array', 10: 'uint64', 11: 'int64',
  12: 'float64',
};

class Reader {
  constructor(buffer) {
    this.buf = buffer;
    this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    this.pos = 0;
  }

  u8() { const v = this.view.getUint8(this.pos); this.pos += 1; return v; }
  u16() { const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  u32() { const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  i32() { const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  u64() { const v = Number(this.view.getBigUint64(this.pos, true)); this.pos += 8; return v; }
  i64() { const v = Number(this.view.getBigInt64(this.pos, true)); this.pos += 8; return v; }
  f32() { const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  f64() { const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }

  string() {
    const len = this.u64();
    const str = this.buf.subarray(this.pos, this.pos + len).toString('utf-8');
    this.pos += len;
    return str;
  }

  value(type) {
    switch (type) {
      case 0: return this.u8();
      case 1: return this.u8(); // int8 read as uint8
      case 2: return this.u16();
      case 3: return this.u16();
      case 4: return this.u32();
      case 5: return this.i32();
      case 6: return this.f32();
      case 7: return Boolean(this.u8());
      case 8: return this.string();
      case 9: {
        const elemType = this.u32();
        const count = this.u64();
        const arr = [];
        for (let i = 0; i < count && i < 256; i++) arr.push(this.value(elemType));
        if (count > 256) this.pos += 0; // skip rest (we only preview)
        return arr;
      }
      case 10: return this.u64();
      case 11: return this.i64();
      case 12: return this.f64();
      default: return null;
    }
  }
}

/**
 * @param {Buffer} buffer
 * @returns {{ metadata: object, tensors: object[], graph: object }}
 */
export function parse(buffer) {
  const r = new Reader(buffer);

  const magic = r.u32();
  if (magic !== GGUF_MAGIC) {
    throw new Error('Not a valid GGUF file');
  }

  const version = r.u32();
  if (version < 2 || version > 3) {
    throw new Error(`Unsupported GGUF version: ${version}`);
  }

  const tensorCount = r.u64();
  const kvCount = r.u64();

  // read metadata
  const metadata = {};
  for (let i = 0; i < kvCount; i++) {
    const key = r.string();
    const valueType = r.u32();
    const val = r.value(valueType);
    metadata[key] = val;
  }

  // read tensor info
  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = r.string();
    const nDims = r.u32();
    const shape = [];
    for (let d = 0; d < nDims; d++) shape.push(r.u64());
    const typeIndex = r.u32();
    const offset = r.u64();

    tensors.push({
      name,
      shape,
      dtype: GGML_TYPES[typeIndex] || `type_${typeIndex}`,
      offset,
    });
  }

  const graph = buildGraph(tensors, metadata);

  return { metadata, tensors, graph };
}

function buildGraph(tensors, metadata) {
  const layers = new Map();

  for (const t of tensors) {
    const parts = t.name.split('.');
    const layerKey = parts.length > 1 ? parts.slice(0, -1).join('.') : t.name;
    const param = parts.length > 1 ? parts[parts.length - 1] : 'data';

    if (!layers.has(layerKey)) {
      layers.set(layerKey, { name: layerKey, type: guessType(layerKey), params: [] });
    }
    layers.get(layerKey).params.push({
      name: param,
      dtype: t.dtype,
      shape: t.shape,
    });
  }

  const nodes = [...layers.values()].map((layer, i) => ({
    id: `n${i}`,
    name: layer.name,
    type: layer.type,
    params: layer.params,
  }));

  const edges = [];
  for (let i = 1; i < nodes.length; i++) {
    edges.push({ from: nodes[i - 1].id, to: nodes[i].id });
  }

  return { nodes, edges };
}

function guessType(name) {
  const lower = name.toLowerCase();
  if (lower.includes('embed') || lower.includes('token')) return 'Embedding';
  if (lower.includes('attn') || lower.includes('attention')) return 'Attention';
  if (lower.includes('mlp') || lower.includes('ffn') || lower.includes('feed_forward')) return 'MLP';
  if (lower.includes('norm') || lower.includes('ln')) return 'Normalization';
  if (lower.includes('output') || lower.includes('lm_head')) return 'Output';
  if (lower.includes('conv')) return 'Convolution';
  return 'Linear';
}
