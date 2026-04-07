import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parse } from '../parsers/gguf.js';

describe('gguf parser', () => {
  function makeGguf({ metadata = {}, tensors = [] } = {}) {
    const parts = [];

    // magic "GGUF"
    const header = Buffer.alloc(24);
    header.writeUInt32LE(0x46554747, 0); // magic
    header.writeUInt32LE(3, 4); // version 3
    header.writeBigUInt64LE(BigInt(tensors.length), 8); // tensor count
    header.writeBigUInt64LE(BigInt(Object.keys(metadata).length), 16); // kv count
    parts.push(header);

    // metadata key-value pairs
    for (const [key, val] of Object.entries(metadata)) {
      // string key
      const keyBuf = Buffer.from(key, 'utf-8');
      const keyLen = Buffer.alloc(8);
      keyLen.writeBigUInt64LE(BigInt(keyBuf.length));
      parts.push(keyLen, keyBuf);

      // type = uint32 (4) + value
      const typeBuf = Buffer.alloc(4);
      typeBuf.writeUInt32LE(4); // uint32
      parts.push(typeBuf);

      const valBuf = Buffer.alloc(4);
      valBuf.writeUInt32LE(val);
      parts.push(valBuf);
    }

    // tensor info
    for (const t of tensors) {
      const nameBuf = Buffer.from(t.name, 'utf-8');
      const nameLen = Buffer.alloc(8);
      nameLen.writeBigUInt64LE(BigInt(nameBuf.length));
      parts.push(nameLen, nameBuf);

      const dims = Buffer.alloc(4);
      dims.writeUInt32LE(t.shape.length);
      parts.push(dims);

      for (const d of t.shape) {
        const dimBuf = Buffer.alloc(8);
        dimBuf.writeBigUInt64LE(BigInt(d));
        parts.push(dimBuf);
      }

      const typeBuf = Buffer.alloc(4);
      typeBuf.writeUInt32LE(t.type ?? 0); // F32
      parts.push(typeBuf);

      const offsetBuf = Buffer.alloc(8);
      offsetBuf.writeBigUInt64LE(BigInt(t.offset ?? 0));
      parts.push(offsetBuf);
    }

    return Buffer.concat(parts);
  }

  it('should parse a valid GGUF file', () => {
    const buf = makeGguf({
      metadata: { 'general.architecture': 7 },
      tensors: [
        { name: 'blk.0.attn.weight', shape: [4096, 4096], type: 0, offset: 0 },
        { name: 'blk.0.ffn.weight', shape: [4096, 11008], type: 0, offset: 67108864 },
      ],
    });

    const result = parse(buf);
    assert.equal(result.tensors.length, 2);
    assert.equal(result.tensors[0].name, 'blk.0.attn.weight');
    assert.deepEqual(result.tensors[0].shape, [4096, 4096]);
  });

  it('should build graph with correct types', () => {
    const buf = makeGguf({
      tensors: [
        { name: 'token_embd.weight', shape: [32000, 4096] },
        { name: 'blk.0.attn.weight', shape: [4096, 4096] },
        { name: 'blk.0.ffn.weight', shape: [4096, 11008] },
        { name: 'output_norm.weight', shape: [4096] },
      ],
    });

    const result = parse(buf);
    assert.equal(result.graph.nodes.length, 4);
    assert.equal(result.graph.nodes[0].type, 'Embedding');
    assert.equal(result.graph.nodes[1].type, 'Attention');
    assert.equal(result.graph.nodes[2].type, 'MLP');
    assert.equal(result.graph.nodes[3].type, 'Normalization');
  });

  it('should reject invalid magic', () => {
    const buf = Buffer.alloc(24);
    buf.writeUInt32LE(0x00000000, 0);
    assert.throws(() => parse(buf), /Not a valid GGUF file/);
  });

  it('should reject unsupported version', () => {
    const buf = Buffer.alloc(24);
    buf.writeUInt32LE(0x46554747, 0);
    buf.writeUInt32LE(1, 4); // version 1
    assert.throws(() => parse(buf), /Unsupported GGUF version/);
  });
});
