import { describe, it, after } from 'node:test';
import assert from 'node:assert/strict';
import { startServer } from '../server.js';

describe('server', () => {
  const mockModel = {
    metadata: { irVersion: 9 },
    tensors: [{ name: 'weight', shape: [10, 10], dtype: 'FLOAT' }],
    graph: {
      nodes: [{ id: 'n0', name: 'relu', type: 'Relu' }],
      edges: [],
    },
  };

  it('should start and serve model data', async () => {
    const { url, close } = await startServer(mockModel, { port: 9990, fileName: 'test.onnx' });
    after(() => close());

    assert.ok(url.includes('9990'));

    const res = await fetch(`${url}/api/model`);
    const data = await res.json();

    assert.equal(data.fileName, 'test.onnx');
    assert.equal(data.graph.nodes.length, 1);
    assert.equal(data.tensors.length, 1);
  });

  it('should serve index.html on /', async () => {
    const { url, close } = await startServer(mockModel, { port: 9991 });
    after(() => close());

    const res = await fetch(`${url}/`);
    const text = await res.text();

    assert.ok(text.includes('mlpeel'));
    assert.equal(res.headers.get('content-type'), 'text/html');
  });

  it('should return 404 for unknown paths', async () => {
    const { url, close } = await startServer(mockModel, { port: 9992 });
    after(() => close());

    const res = await fetch(`${url}/nonexistent`);
    assert.equal(res.status, 404);
  });

  it('should auto-increment port on conflict', async () => {
    const { url: url1, close: close1 } = await startServer(mockModel, { port: 9993 });
    const { url: url2, close: close2 } = await startServer(mockModel, { port: 9993 });
    after(() => { close1(); close2(); });

    assert.ok(url1.includes('9993'));
    assert.ok(url2.includes('9994'));
  });
});
