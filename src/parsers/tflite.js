/**
 * TFLite parser.
 *
 * TFLite uses FlatBuffers binary format.
 * File identifier: "TFL3" at bytes 4-7.
 *
 * We decode the minimal schema: Model -> SubGraphs -> Tensors/Operators.
 */

const TENSOR_TYPES = [
  'FLOAT32', 'FLOAT16', 'INT32', 'UINT8', 'INT64', 'STRING',
  'BOOL', 'INT16', 'COMPLEX64', 'INT8', 'FLOAT64', 'COMPLEX128',
  'UINT64', 'RESOURCE', 'VARIANT', 'UINT32', 'UINT16', 'INT4',
];

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs
const BUILTIN_OPS = [
  'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D',
  'DEPTH_TO_SPACE', 'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED',
  'HASHTABLE_LOOKUP', 'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION',
  'LOGISTIC', 'LSH_PROJECTION', 'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU',
  'RELU_N1_TO_1', 'RELU6', 'RESHAPE', 'RESIZE_BILINEAR', 'RNN', 'SOFTMAX',
  'SPACE_TO_DEPTH', 'SVDF', 'TANH', 'CONCAT_EMBEDDINGS', 'SKIP_GRAM',
  'CALL', 'CUSTOM', 'EMBEDDING_LOOKUP_SPARSE', 'PAD', 'UNIDIRECTIONAL_SEQUENCE_RNN',
  'GATHER', 'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND', 'TRANSPOSE',
  'MEAN', 'SUB', 'DIV', 'SQUEEZE', 'UNIDIRECTIONAL_SEQUENCE_LSTM',
  'STRIDED_SLICE', 'BIDIRECTIONAL_SEQUENCE_RNN', 'EXP', 'TOPK_V2',
  'SPLIT', 'LOG_SOFTMAX', 'DELEGATE', 'BIDIRECTIONAL_SEQUENCE_LSTM',
  'CAST', 'PRELU', 'MAXIMUM', 'ARG_MAX', 'MINIMUM', 'LESS',
  'NEG', 'PADV2', 'GREATER', 'GREATER_EQUAL', 'LESS_EQUAL',
  'SELECT', 'SLICE', 'SIN', 'TRANSPOSE_CONV', 'SPARSE_TO_DENSE',
  'TILE', 'EXPAND_DIMS', 'EQUAL', 'NOT_EQUAL', 'LOG', 'SUM',
  'SQRT', 'RSQRT', 'SHAPE', 'POW', 'ARG_MIN', 'FAKE_QUANT',
  'REDUCE_PROD', 'REDUCE_MAX', 'PACK', 'LOGICAL_OR', 'ONE_HOT',
  'LOGICAL_AND', 'LOGICAL_NOT', 'UNPACK', 'REDUCE_MIN', 'FLOOR_DIV',
  'REDUCE_ANY', 'SQUARE', 'ZEROS_LIKE', 'FILL', 'FLOOR_MOD',
  'RANGE', 'RESIZE_NEAREST_NEIGHBOR', 'LEAKY_RELU', 'SQUARED_DIFFERENCE',
  'MIRROR_PAD', 'ABS', 'SPLIT_V', 'UNIQUE', 'CEIL', 'REVERSE_V2',
  'ADD_N', 'GATHER_ND', 'COS', 'WHERE', 'RANK', 'ELU',
  'REVERSE_SEQUENCE', 'MATRIX_DIAG', 'QUANTIZE', 'MATRIX_SET_DIAG',
  'ROUND', 'HARD_SWISH', 'IF', 'WHILE', 'NON_MAX_SUPPRESSION_V4',
  'NON_MAX_SUPPRESSION_V5', 'SCATTER_ND', 'SELECT_V2', 'DENSIFY',
  'SEGMENT_SUM', 'BATCH_MATMUL', 'PLACEHOLDER_FOR_GREATER_OP_CODES',
  'CUMSUM', 'CALL_ONCE', 'BROADCAST_TO', 'RFFT2D', 'CONV_3D',
  'IMAG', 'REAL', 'COMPLEX_ABS', 'HASHTABLE', 'HASHTABLE_FIND',
  'HASHTABLE_IMPORT', 'HASHTABLE_SIZE', 'REDUCE_ALL', 'CONV_3D_TRANSPOSE',
  'VAR_HANDLE', 'READ_VARIABLE', 'ASSIGN_VARIABLE', 'BROADCAST_ARGS',
  'RANDOM_STANDARD_NORMAL', 'BUCKETIZE', 'RANDOM_UNIFORM', 'MULTINOMIAL',
  'GELU', 'DYNAMIC_UPDATE_SLICE', 'RELU_0_TO_1', 'UNSORTED_SEGMENT_PROD',
  'UNSORTED_SEGMENT_MAX', 'UNSORTED_SEGMENT_SUM', 'ATAN2', 'UNSORTED_SEGMENT_MIN',
  'SIGN', 'BITCAST', 'BITWISE_XOR', 'RIGHT_SHIFT', 'STABLEHLO_SCATTER',
  'DILATE', 'STABLEHLO_RNG_BIT_GENERATOR', 'REDUCE_WINDOW',
];

/**
 * Minimal FlatBuffer reader.
 */
class FBReader {
  constructor(buf) {
    this.buf = buf;
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  u8(off) { return this.view.getUint8(off); }
  u16(off) { return this.view.getUint16(off, true); }
  i16(off) { return this.view.getInt16(off, true); }
  u32(off) { return this.view.getUint32(off, true); }
  i32(off) { return this.view.getInt32(off, true); }

  // read root table offset
  root() { return this.u32(0); }

  // read vtable for a table at `tableOff`
  vtable(tableOff) {
    const soff = this.i32(tableOff);
    return tableOff - soff;
  }

  // get field offset within table (fieldIndex is 0-based)
  fieldOff(tableOff, fieldIndex) {
    const vt = this.vtable(tableOff);
    const vtSize = this.u16(vt);
    const slotOff = 4 + fieldIndex * 2;
    if (slotOff >= vtSize) return 0;
    const off = this.u16(vt + slotOff);
    return off ? tableOff + off : 0;
  }

  // read scalar field
  fieldU8(tableOff, fieldIndex, def = 0) {
    const off = this.fieldOff(tableOff, fieldIndex);
    return off ? this.u8(off) : def;
  }

  fieldU32(tableOff, fieldIndex, def = 0) {
    const off = this.fieldOff(tableOff, fieldIndex);
    return off ? this.u32(off) : def;
  }

  fieldI32(tableOff, fieldIndex, def = 0) {
    const off = this.fieldOff(tableOff, fieldIndex);
    return off ? this.i32(off) : def;
  }

  // read string field
  fieldStr(tableOff, fieldIndex) {
    const off = this.fieldOff(tableOff, fieldIndex);
    if (!off) return '';
    const strOff = off + this.u32(off);
    const len = this.u32(strOff);
    return this.buf.subarray(strOff + 4, strOff + 4 + len).toString('utf-8');
  }

  // read vector field, returns { offset, length } or null
  fieldVec(tableOff, fieldIndex) {
    const off = this.fieldOff(tableOff, fieldIndex);
    if (!off) return null;
    const vecOff = off + this.u32(off);
    const len = this.u32(vecOff);
    return { offset: vecOff + 4, length: len };
  }

  // read table from a vector of tables
  vecTable(vec, index) {
    const elemOff = vec.offset + index * 4;
    return elemOff + this.u32(elemOff);
  }

  // read i32 from a vector of scalars
  vecI32(vec, index) {
    return this.i32(vec.offset + index * 4);
  }
}

/**
 * @param {Buffer} buffer
 * @returns {{ metadata: object, tensors: object[], graph: object }}
 */
export function parse(buffer) {
  const fb = new FBReader(buffer);

  // verify file identifier
  const ident = buffer.subarray(4, 8).toString('ascii');
  if (ident !== 'TFL3') {
    throw new Error(`Not a valid TFLite file (identifier: "${ident}")`);
  }

  const modelOff = fb.root();

  // Model fields: 0=version, 1=operator_codes, 2=subgraphs, 3=description, 4=buffers
  const version = fb.fieldU32(modelOff, 0, 0);
  const description = fb.fieldStr(modelOff, 3);

  // read operator codes
  const opCodesVec = fb.fieldVec(modelOff, 1);
  const opCodes = [];
  if (opCodesVec) {
    for (let i = 0; i < opCodesVec.length; i++) {
      const ocOff = fb.vecTable(opCodesVec, i);
      // field 0: deprecated_builtin_code (int8), field 1: custom_code (string), field 6: builtin_code (int32)
      const deprecatedCode = fb.fieldI32(ocOff, 0, 0) & 0xff;
      const customCode = fb.fieldStr(ocOff, 1);
      const builtinCode = fb.fieldI32(ocOff, 6, -1);
      // use builtin_code if available, fall back to deprecated
      const code = builtinCode >= 0 ? builtinCode : deprecatedCode;
      opCodes.push({ code, customCode });
    }
  }

  // read subgraphs
  const subgraphsVec = fb.fieldVec(modelOff, 2);
  const subgraphs = [];

  if (subgraphsVec) {
    for (let s = 0; s < subgraphsVec.length; s++) {
      const sgOff = fb.vecTable(subgraphsVec, s);
      subgraphs.push(readSubGraph(fb, sgOff, opCodes));
    }
  }

  const metadata = { version, description, subgraphCount: subgraphs.length };

  // use first subgraph as the main graph
  const main = subgraphs[0] || { tensors: [], operators: [], inputs: [], outputs: [] };
  const tensors = main.tensors;
  const graph = buildGraph(main, opCodes);

  return { metadata, tensors, graph };
}

function readSubGraph(fb, sgOff, opCodes) {
  // SubGraph fields: 0=tensors, 1=inputs, 2=outputs, 3=operators, 4=name
  const name = fb.fieldStr(sgOff, 4);

  // tensors
  const tensorsVec = fb.fieldVec(sgOff, 0);
  const tensors = [];
  if (tensorsVec) {
    for (let i = 0; i < tensorsVec.length; i++) {
      const tOff = fb.vecTable(tensorsVec, i);
      // Tensor fields: 0=shape, 1=type, 2=buffer, 3=name
      const shapeVec = fb.fieldVec(tOff, 0);
      const shape = [];
      if (shapeVec) {
        for (let d = 0; d < shapeVec.length; d++) shape.push(fb.vecI32(shapeVec, d));
      }
      const typeIdx = fb.fieldU8(tOff, 1, 0);
      const bufferIdx = fb.fieldU32(tOff, 2, 0);
      const tName = fb.fieldStr(tOff, 3);

      tensors.push({
        name: tName,
        shape,
        dtype: TENSOR_TYPES[typeIdx] || `type_${typeIdx}`,
        bufferIndex: bufferIdx,
      });
    }
  }

  // inputs/outputs (vectors of int32 tensor indices)
  const inputsVec = fb.fieldVec(sgOff, 1);
  const inputs = [];
  if (inputsVec) {
    for (let i = 0; i < inputsVec.length; i++) inputs.push(fb.vecI32(inputsVec, i));
  }

  const outputsVec = fb.fieldVec(sgOff, 2);
  const outputs = [];
  if (outputsVec) {
    for (let i = 0; i < outputsVec.length; i++) outputs.push(fb.vecI32(outputsVec, i));
  }

  // operators
  const opsVec = fb.fieldVec(sgOff, 3);
  const operators = [];
  if (opsVec) {
    for (let i = 0; i < opsVec.length; i++) {
      const opOff = fb.vecTable(opsVec, i);
      // Operator fields: 0=opcode_index, 1=inputs, 2=outputs
      const opcodeIndex = fb.fieldU32(opOff, 0, 0);
      const opInputsVec = fb.fieldVec(opOff, 1);
      const opOutputsVec = fb.fieldVec(opOff, 2);

      const opInputs = [];
      if (opInputsVec) {
        for (let j = 0; j < opInputsVec.length; j++) opInputs.push(fb.vecI32(opInputsVec, j));
      }
      const opOutputs = [];
      if (opOutputsVec) {
        for (let j = 0; j < opOutputsVec.length; j++) opOutputs.push(fb.vecI32(opOutputsVec, j));
      }

      operators.push({ opcodeIndex, inputs: opInputs, outputs: opOutputs });
    }
  }

  return { name, tensors, inputs, outputs, operators };
}

function buildGraph(subgraph, opCodes) {
  const { tensors, operators, inputs: sgInputs, outputs: sgOutputs } = subgraph;

  // map: tensor index -> which node produces it
  const producerMap = new Map();

  // operator nodes
  const nodes = operators.map((op, i) => {
    const id = `n${i}`;
    const oc = opCodes[op.opcodeIndex] || {};
    const opName = oc.customCode || BUILTIN_OPS[oc.code] || `OP_${oc.code}`;

    for (const outIdx of op.outputs) producerMap.set(outIdx, id);

    // separate weight tensors from activation inputs
    const paramIndices = op.inputs.filter(idx =>
      idx >= 0 && !sgInputs.includes(idx) && !producerMap.has(idx)
    );
    const activationInputs = op.inputs.filter(idx =>
      idx >= 0 && (sgInputs.includes(idx) || producerMap.has(idx))
    );

    return {
      id,
      name: tensors[op.outputs[0]]?.name || `${opName}_${i}`,
      type: opName,
      inputs: activationInputs.map(idx => tensors[idx]?.name || `tensor_${idx}`),
      outputs: op.outputs.map(idx => tensors[idx]?.name || `tensor_${idx}`),
      params: paramIndices.map(idx => {
        const t = tensors[idx];
        return t ? { name: t.name?.split('/').pop() || `tensor_${idx}`, dtype: t.dtype, shape: t.shape } : null;
      }).filter(Boolean),
    };
  });

  // edges between operator nodes
  const edges = [];
  for (const op of operators) {
    const toId = nodes.find(n => n.id === `n${operators.indexOf(op)}`)?.id;
    for (const inIdx of op.inputs) {
      if (inIdx < 0) continue;
      const fromId = producerMap.get(inIdx);
      if (fromId && fromId !== toId) {
        edges.push({ from: fromId, to: toId });
      }
    }
  }

  // input nodes
  const inputNodes = sgInputs.map((idx, i) => {
    const t = tensors[idx];
    return {
      id: `input_${i}`,
      name: t?.name || `input_${i}`,
      type: 'Input',
      params: t ? [{ name: 'type', dtype: t.dtype, shape: t.shape }] : [],
    };
  });

  // connect inputs to first operators that use them
  for (const idx of sgInputs) {
    const inputNode = inputNodes.find(n => n.name === tensors[idx]?.name);
    if (!inputNode) continue;
    for (let i = 0; i < operators.length; i++) {
      if (operators[i].inputs.includes(idx)) {
        edges.push({ from: inputNode.id, to: `n${i}` });
      }
    }
  }

  // output nodes
  const outputNodes = sgOutputs.map((idx, i) => {
    const t = tensors[idx];
    return {
      id: `output_${i}`,
      name: t?.name || `output_${i}`,
      type: 'Output',
      params: t ? [{ name: 'type', dtype: t.dtype, shape: t.shape }] : [],
    };
  });

  for (const idx of sgOutputs) {
    const fromId = producerMap.get(idx);
    const outNode = outputNodes.find(n => n.name === tensors[idx]?.name);
    if (fromId && outNode) edges.push({ from: fromId, to: outNode.id });
  }

  return {
    nodes: [...inputNodes, ...nodes, ...outputNodes],
    edges,
  };
}
