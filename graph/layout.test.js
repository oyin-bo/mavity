// @ts-check

import assert from 'node:assert';
import { test } from 'node:test';


import { GraphLayout } from './layout.js';

import { webGL2 } from 'webgl2';
import { createTextureR32F, createTextureRGBA32F } from './utils.js';

test('GraphLayout: constructor validates missing textures', async () => {
  const gl = await getGL();

  // @ts-ignore
  assert.throws(() => new GraphLayout({
    gl,
    particleCount: 128,
    edgeCount: 128
  }), /Missing required textures/);

  resetGL();
});

test('GraphLayout: basic allocation and single-frame run', async () => {
  const gl = await getGL();

  const particleCount = 10;
  const edgeCount = 20;
  const particleDataWidth = 4;
  const particleDataHeight = 4;


  // Mock textures
  const texPosition = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const texVelocity = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const texIdMassTint = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const texEdgePtr = createTextureR32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const texEdgeStore = createTextureR32F({ gl, width: particleDataWidth, height: particleDataHeight });

  const layout = new GraphLayout({
    gl,
    particleCount,
    edgeCount,
    texPosition,
    texVelocity,
    texIdMassTint,
    texEdgePtr,
    texEdgeStore
  });

  assert.deepStrictEqual(
    {
      particleCount: layout.particleCount,
      edgeCount: layout.edgeCount,
      renderCount: layout.renderCount
    },
    {
      particleCount,
      edgeCount,
      renderCount: 0
    });

  // Run one frame
  layout.run();

  assert.deepStrictEqual(
    {
      renderCount: layout.renderCount,
      frameCounter: layout.frameCounter,
      passCounter: layout.passCounter
    },
    {
      renderCount: 1,
      frameCounter: 1,
      passCounter: 1
    });

  // Verify reflection
  const snapshot = layout.valueOf({ pixels: false });
  assert.deepStrictEqual(
    {
      renderCount: snapshot.renderCount,
      position: snapshot.position ? 'Position snapshot should exist' : snapshot.position,
      toString: snapshot.toString().includes('GraphLayout') ? 'toString should contain class name' : snapshot.toString()
    },
    {
      renderCount: 1,
      position: 'Position snapshot should exist',
      toString: 'toString should contain class name'
  });

  layout.dispose();
  // Textures are removed by layout.dispose(); avoid double-delete
  resetGL();
});

async function getGL() {
  const gl = await webGL2({ debug: true });
  return gl;
}

function resetGL() {
}