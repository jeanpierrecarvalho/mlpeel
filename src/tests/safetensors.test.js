import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parse } from '../parsers/safetensors.js';

describe('safetensors parser', () => {
  function makeSafetensors(header) {
    const json = JSON.stringify(header);
    const jsonBuf = Buffer.from(json, 'utf-8');
    const lenBuf = Buffer.alloc(8);
    lenBuf.writeBigUInt64LE(BigInt(jsonBuf.length));
    return Buffer.concat([lenBuf, jsonBuf]);
  }

  it('should parse a simple safetensors file', () => {
    const header = {
      'model.layer.0.weight': {
        dtype: 'F32',
        shape: [768, 768],
        data_offsets: [0, 2359296],
      },
      'model.layer.0.bias': {
        dtype: 'F32',
        shape: [768],
        data_offsets: [2359296, 2362368],
      },
    };

    const buf = makeSafetensors(header);
    const result = parse(buf);

    assert.equal(result.tensors.length, 2);
    assert.equal(result.tensors[0].name, 'model.layer.0.weight');
    assert.deepEqual(result.tensors[0].shape, [768, 768]);
    assert.equal(result.tensors[0].dtype, 'F32');
  });

  it('should extract metadata', () => {
    const header = {
      __metadata__: { format: 'pt', source: 'huggingface' },
      'weight': { dtype: 'F16', shape: [10], data_offsets: [0, 20] },
    };

    const buf = makeSafetensors(header);
    const result = parse(buf);

    assert.equal(result.metadata.format, 'pt');
    assert.equal(result.tensors.length, 1);
  });

  it('should build graph from tensor names', () => {
    const header = {
      'model.embed.weight': { dtype: 'F32', shape: [50000, 768], data_offsets: [0, 100] },
      'model.layers.0.attn.weight': { dtype: 'F32', shape: [768, 768], data_offsets: [100, 200] },
      'model.layers.0.mlp.weight': { dtype: 'F32', shape: [768, 3072], data_offsets: [200, 300] },
      'model.norm.weight': { dtype: 'F32', shape: [768], data_offsets: [300, 400] },
    };

    const buf = makeSafetensors(header);
    const result = parse(buf);

    assert.equal(result.graph.nodes.length, 4);
    assert.equal(result.graph.nodes[0].type, 'Embedding');
    assert.equal(result.graph.nodes[1].type, 'Attention');
    assert.equal(result.graph.nodes[2].type, 'MLP');
    assert.equal(result.graph.nodes[3].type, 'Normalization');
    assert.equal(result.graph.edges.length, 3);
  });

  it('should reject invalid header length', () => {
    const buf = Buffer.alloc(16);
    buf.writeBigUInt64LE(BigInt(9999999));
    assert.throws(() => parse(buf), /header length exceeds file size/);
  });
});
