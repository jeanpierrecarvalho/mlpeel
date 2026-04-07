/**
 * ONNX parser.
 *
 * ONNX uses Protocol Buffers (protobuf) wire format.
 * This is a minimal decoder — no .proto schema needed,
 * just field-number mappings for the structures we care about.
 */

const VARINT = 0;
const FIXED64 = 1;
const LEN = 2;
const FIXED32 = 5;

class ProtoReader {
  constructor(buffer) {
    this.buf = buffer;
    this.pos = 0;
    this.end = buffer.length;
  }

  done() { return this.pos >= this.end; }

  varint() {
    let result = 0;
    let shift = 0;
    while (this.pos < this.end) {
      const byte = this.buf[this.pos++];
      result += (byte & 0x7f) * (2 ** shift);
      if ((byte & 0x80) === 0) return result;
      shift += 7;
      if (shift > 63) break;
    }
    return result;
  }

  fixed32() {
    const v = this.buf.readUInt32LE(this.pos);
    this.pos += 4;
    return v;
  }

  float32() {
    const v = this.buf.readFloatLE(this.pos);
    this.pos += 4;
    return v;
  }

  fixed64() {
    const lo = this.buf.readUInt32LE(this.pos);
    const hi = this.buf.readUInt32LE(this.pos + 4);
    this.pos += 8;
    return lo + hi * 0x100000000;
  }

  bytes() {
    const len = this.varint();
    const data = this.buf.subarray(this.pos, this.pos + len);
    this.pos += len;
    return data;
  }

  string() { return this.bytes().toString('utf-8'); }

  skip(wireType) {
    switch (wireType) {
      case VARINT: this.varint(); break;
      case FIXED64: this.pos += 8; break;
      case LEN: this.pos += this.varint(); break;
      case 3: // start group (deprecated)
        while (!this.done()) {
          const t = this.tag();
          if (t.wire === 4) break;
          this.skip(t.wire);
        }
        break;
      case 4: break; // end group
      case FIXED32: this.pos += 4; break;
      default: this.pos = this.end; break; // bail out on unknown
    }
  }

  tag() {
    const v = this.varint();
    return { field: Math.floor(v / 8), wire: v % 8 };
  }
}

const DATA_TYPES = [
  'UNDEFINED', 'FLOAT', 'UINT8', 'INT8', 'UINT16', 'INT16',
  'INT32', 'INT64', 'STRING', 'BOOL', 'FLOAT16', 'DOUBLE',
  'UINT32', 'UINT64', 'COMPLEX64', 'COMPLEX128', 'BFLOAT16',
];

function decodeShape(data) {
  const dims = [];
  const r = new ProtoReader(data);
  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) {
      const dimR = new ProtoReader(r.bytes());
      let value = null;
      let param = null;
      while (!dimR.done()) {
        const dt = dimR.tag();
        if (dt.field === 1 && dt.wire === VARINT) value = dimR.varint();
        else if (dt.field === 2 && dt.wire === LEN) param = dimR.string();
        else dimR.skip(dt.wire);
      }
      dims.push(param || value);
    } else {
      r.skip(wire);
    }
  }
  return dims;
}

function decodeTypeProto(data) {
  const r = new ProtoReader(data);
  let dtype = null;
  let shape = null;
  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) {
      const tr = new ProtoReader(r.bytes());
      while (!tr.done()) {
        const tt = tr.tag();
        if (tt.field === 1 && tt.wire === VARINT) dtype = DATA_TYPES[tr.varint()] || 'UNKNOWN';
        else if (tt.field === 2 && tt.wire === LEN) shape = decodeShape(tr.bytes());
        else tr.skip(tt.wire);
      }
    } else {
      r.skip(wire);
    }
  }
  return { dtype, shape };
}

function decodeValueInfo(data) {
  const r = new ProtoReader(data);
  let name = '';
  let type = null;
  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) name = r.string();
    else if (field === 2 && wire === LEN) type = decodeTypeProto(r.bytes());
    else r.skip(wire);
  }
  return { name, ...type };
}

// ONNX AttributeProto fields:
//  1: name (string)      2: f (float)       3: i (int64)
//  4: s (bytes)          5: t (TensorProto)  6: g (GraphProto)
//  7: floats (packed)    8: ints (packed)    20: type (enum)
function decodeAttribute(data) {
  const r = new ProtoReader(data);
  let name = '';
  let value = null;
  let type = 0;
  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) name = r.string();
    else if (field === 2 && wire === FIXED32) value = r.float32();
    else if (field === 3 && wire === VARINT) value = r.varint();
    else if (field === 4 && wire === LEN) value = r.bytes().toString('utf-8');
    else if (field === 7 && wire === LEN) {
      // packed floats
      const packed = r.bytes();
      const floats = [];
      for (let i = 0; i < packed.length; i += 4) floats.push(packed.readFloatLE(i));
      value = floats;
    }
    else if (field === 8 && wire === LEN) {
      // packed ints
      const pr = new ProtoReader(r.bytes());
      const ints = [];
      while (!pr.done()) ints.push(pr.varint());
      value = ints;
    }
    else if (field === 8 && wire === VARINT) {
      // unpacked repeated int
      if (!Array.isArray(value)) value = [];
      value.push(r.varint());
    }
    else if (field === 20 && wire === VARINT) type = r.varint();
    else r.skip(wire);
  }
  return { name, value, type };
}

// ONNX NodeProto fields:
//  1: input (string)  2: output (string)  3: name (string)
//  4: op_type (string)  5: attribute (AttributeProto)
//  6: doc_string (string)  7: domain (string)
function decodeNode(data) {
  const r = new ProtoReader(data);
  const inputs = [];
  const outputs = [];
  let name = '';
  let opType = '';
  const attributes = [];

  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) inputs.push(r.string());
    else if (field === 2 && wire === LEN) outputs.push(r.string());
    else if (field === 3 && wire === LEN) name = r.string();
    else if (field === 4 && wire === LEN) opType = r.string();
    else if (field === 5 && wire === LEN) attributes.push(decodeAttribute(r.bytes()));
    else r.skip(wire);
  }

  return { name, opType, inputs, outputs, attributes };
}

// ONNX TensorProto fields:
//  1: dims (int64)  2: data_type (int32)  4: float_data (packed)
//  8: name (string)  13: raw_data (bytes)
function decodeTensor(data) {
  const r = new ProtoReader(data);
  const dims = [];
  let name = '';
  let dtype = 0;
  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === VARINT) dims.push(r.varint());
    else if (field === 1 && wire === LEN) {
      // packed dims
      const pr = new ProtoReader(r.bytes());
      while (!pr.done()) dims.push(pr.varint());
    }
    else if (field === 2 && wire === VARINT) dtype = r.varint();
    else if (field === 8 && wire === LEN) name = r.string();
    else r.skip(wire);
  }
  return { name, shape: dims, dtype: DATA_TYPES[dtype] || 'UNKNOWN' };
}

// ONNX GraphProto fields:
//  1: node  2: name  5: initializer  11: input  12: output
function decodeGraph(data) {
  const r = new ProtoReader(data);
  const nodes = [];
  const inputs = [];
  const outputs = [];
  const initializers = [];
  let name = '';

  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === LEN) nodes.push(decodeNode(r.bytes()));
    else if (field === 2 && wire === LEN) name = r.string();
    else if (field === 5 && wire === LEN) initializers.push(decodeTensor(r.bytes()));
    else if (field === 11 && wire === LEN) inputs.push(decodeValueInfo(r.bytes()));
    else if (field === 12 && wire === LEN) outputs.push(decodeValueInfo(r.bytes()));
    else r.skip(wire);
  }

  return { name, nodes, inputs, outputs, initializers };
}

/**
 * @param {Buffer} buffer
 * @returns {{ metadata: object, tensors: object[], graph: object }}
 */
export function parse(buffer) {
  const r = new ProtoReader(buffer);
  let irVersion = 0;
  const opsetImports = [];
  let graph = null;
  let producerName = '';
  let producerVersion = '';
  let domain = '';
  let modelVersion = 0;
  let docString = '';

  while (!r.done()) {
    const { field, wire } = r.tag();
    if (field === 1 && wire === VARINT) irVersion = r.varint();
    else if (field === 2 && wire === LEN) producerName = r.string();
    else if (field === 3 && wire === LEN) producerVersion = r.string();
    else if (field === 4 && wire === LEN) domain = r.string();
    else if (field === 5 && wire === VARINT) modelVersion = r.varint();
    else if (field === 6 && wire === LEN) docString = r.string();
    else if (field === 7 && wire === LEN) graph = decodeGraph(r.bytes());
    else if (field === 8 && wire === LEN) {
      const oR = new ProtoReader(r.bytes());
      let opDomain = '';
      let opVersion = 0;
      while (!oR.done()) {
        const ot = oR.tag();
        if (ot.field === 1 && ot.wire === LEN) opDomain = oR.string();
        else if (ot.field === 2 && ot.wire === VARINT) opVersion = oR.varint();
        else oR.skip(ot.wire);
      }
      opsetImports.push({ domain: opDomain || 'ai.onnx', version: opVersion });
    }
    else r.skip(wire);
  }

  if (!graph) throw new Error('No graph found in ONNX model');

  const metadata = {
    irVersion, producerName, producerVersion,
    domain, modelVersion, docString, opsetImports,
  };

  return {
    metadata,
    tensors: graph.initializers,
    graph: buildGraph(graph),
  };
}

function buildGraph(onnxGraph) {
  const producerMap = new Map();
  const initializerNames = new Set(onnxGraph.initializers.map(i => i.name));

  const nodes = onnxGraph.nodes.map((node, i) => {
    const id = `n${i}`;
    for (const out of node.outputs) producerMap.set(out, id);

    return {
      id,
      name: node.name || `${node.opType}_${i}`,
      type: node.opType,
      inputs: node.inputs.filter(inp => !initializerNames.has(inp)),
      outputs: node.outputs,
      attributes: node.attributes.map(a => ({ name: a.name, value: a.value })),
      params: node.inputs
        .filter(inp => initializerNames.has(inp))
        .map(inp => {
          const init = onnxGraph.initializers.find(t => t.name === inp);
          return init ? { name: inp.split('/').pop(), dtype: init.dtype, shape: init.shape } : null;
        })
        .filter(Boolean),
    };
  });

  const edges = [];
  for (const node of nodes) {
    for (const inp of node.inputs) {
      const fromId = producerMap.get(inp);
      if (fromId) edges.push({ from: fromId, to: node.id, label: inp.split('/').pop() });
    }
  }

  const inputNodes = onnxGraph.inputs
    .filter(v => !initializerNames.has(v.name))
    .map((v, i) => ({
      id: `input_${i}`,
      name: v.name,
      type: 'Input',
      params: [{ name: 'type', dtype: v.dtype, shape: v.shape }],
    }));

  for (const inp of inputNodes) producerMap.set(inp.name, inp.id);

  for (const node of nodes) {
    for (const inp of node.inputs) {
      const fromInput = inputNodes.find(n => n.name === inp);
      if (fromInput) edges.push({ from: fromInput.id, to: node.id });
    }
  }

  const outputNodes = onnxGraph.outputs.map((v, i) => ({
    id: `output_${i}`,
    name: v.name,
    type: 'Output',
    params: [{ name: 'type', dtype: v.dtype, shape: v.shape }],
  }));

  for (const out of outputNodes) {
    const fromId = producerMap.get(out.name);
    if (fromId) edges.push({ from: fromId, to: out.id });
  }

  return {
    nodes: [...inputNodes, ...nodes, ...outputNodes],
    edges,
  };
}
