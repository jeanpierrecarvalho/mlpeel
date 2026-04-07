/**
 * ONNX inference engine.
 * Requires onnxruntime-node (optional dependency).
 */

let ort = null;

async function loadRuntime() {
  if (ort) return ort;
  try {
    ort = await import('onnxruntime-node');
    return ort;
  } catch {
    throw new Error(
      'onnxruntime-node is required for inference.\n' +
      '  Install it with: npm install onnxruntime-node'
    );
  }
}

/**
 * Get input/output metadata from an ONNX model.
 * @param {string} filePath
 * @returns {Promise<{ inputs: object[], outputs: object[] }>}
 */
export async function getIOInfo(filePath) {
  const { InferenceSession } = await loadRuntime();
  const session = await InferenceSession.create(filePath);

  const inputs = session.inputNames.map(name => {
    const meta = session.inputMetadata?.[name];
    return { name, ...(meta || {}) };
  });

  const outputs = session.outputNames.map(name => {
    const meta = session.outputMetadata?.[name];
    return { name, ...(meta || {}) };
  });

  await session.release();
  return { inputs, outputs };
}

const TYPE_MAP = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
};

/**
 * Run inference on an ONNX model.
 * @param {string} filePath - path to .onnx file
 * @param {object} inputData - { inputName: { data: number[], shape: number[], type?: string } }
 * @returns {Promise<object>} - { outputName: { data: number[], shape: number[], type: string } }
 */
export async function runInference(filePath, inputData) {
  const { InferenceSession, Tensor } = await loadRuntime();
  const session = await InferenceSession.create(filePath);

  const feeds = {};
  for (const [name, input] of Object.entries(inputData)) {
    const type = input.type || 'float32';
    const TypedArray = TYPE_MAP[type] || Float32Array;
    const data = new TypedArray(input.data);
    feeds[name] = new Tensor(type, data, input.shape);
  }

  const results = await session.run(feeds);

  const output = {};
  for (const name of session.outputNames) {
    const tensor = results[name];
    output[name] = {
      data: Array.from(tensor.data, v => typeof v === 'bigint' ? Number(v) : v),
      shape: tensor.dims,
      type: tensor.type,
    };
  }

  await session.release();
  return output;
}
