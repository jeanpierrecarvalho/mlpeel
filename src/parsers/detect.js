import { readFile } from 'node:fs/promises';
import { extname } from 'node:path';

const SIGNATURES = {
  onnx: [0x08], // protobuf varint field 1
  gguf: [0x47, 0x47, 0x55, 0x46], // "GGUF"
};

/**
 * Detect model format from file extension and magic bytes.
 * @param {string} filePath
 * @returns {Promise<{ format: string, buffer: Buffer }>}
 */
export async function detect(filePath) {
  const ext = extname(filePath).toLowerCase();
  const buffer = await readFile(filePath);

  if (ext === '.safetensors') {
    return { format: 'safetensors', buffer };
  }

  if (ext === '.gguf') {
    return { format: 'gguf', buffer };
  }

  if (ext === '.tflite') {
    return { format: 'tflite', buffer };
  }

  if (ext === '.onnx') {
    return { format: 'onnx', buffer };
  }

  // fallback: check magic bytes
  const head = buffer.subarray(0, 8);

  if (head[0] === SIGNATURES.gguf[0] && head[1] === SIGNATURES.gguf[1] &&
      head[2] === SIGNATURES.gguf[2] && head[3] === SIGNATURES.gguf[3]) {
    return { format: 'gguf', buffer };
  }

  // TFLite: file identifier "TFL3" at bytes 4-7
  if (head[4] === 0x54 && head[5] === 0x46 && head[6] === 0x4C && head[7] === 0x33) {
    return { format: 'tflite', buffer };
  }

  // ONNX protobuf starts with field 1 (ir_version) varint
  if (head[0] === 0x08) {
    return { format: 'onnx', buffer };
  }

  throw new Error(`Unknown model format: ${filePath}`);
}
