// @ts-check

import assert from 'node:assert';
import { test } from 'node:test';


import { KPhysicsSentinel } from './k-physics-sentinel.js';

import { webGL2 } from 'webgl2';
import { createTextureR32F, createTextureRGBA32F } from './utils.js';
import { readLinear } from '../gravity/diag.js';

test('KPhysicsSentinel: basic allocation', async () => {
  const gl = await getGL();

  const particleDataWidth = 4, particleDataHeight = 4;

  // Mock textures
  const outParticlePositionAndSFC = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outParticleVelocity = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outIdMassTintEdgePtr = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });

  const kSentinel = new KPhysicsSentinel({
    gl,
    particleDataWidth,
    particleDataHeight
  });

  assert.deepStrictEqual(
    {
      particleDataWidth: kSentinel.particleDataHeight,
      particleDataHeight: kSentinel.particleDataHeight,
      renderCount: kSentinel.renderCount
    },
    {
      particleDataWidth,
      particleDataHeight,
      renderCount: 0
    });

  kSentinel.dispose();
  resetGL();
});


test('KPhysicsSentinel: basic allocation and single-frame run', async () => {
  const gl = await getGL();

  const particleDataWidth = 4, particleDataHeight = 4;

  // Mock textures
  const outParticlePositionAndSFC = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outParticleVelocity = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outIdMassTintEdgePtr = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });

  const kSentinel = new KPhysicsSentinel({
    gl,
    particleDataWidth,
    particleDataHeight
  });

  // Run one frame
  kSentinel.run({
    outParticlePositionAndSFC,
    outParticleVelocity,
    outIdMassTintEdgePtr
  });

  // Verify reflection
  assert.deepEqual(
    {
      kSentinel: kSentinel.toString(),
      outParticlePositionAndSFC:
        readLinear({
          gl,
          texture: outParticlePositionAndSFC,
          width: particleDataWidth, height: particleDataHeight,
          count: particleDataWidth * particleDataHeight,
          channels: ['x', 'y', 'z', 'sfc']
        }).toString(),
      outParticleVelocity: readLinear({
        gl,
        texture: outParticleVelocity,
        width: particleDataWidth, height: particleDataHeight,
        count: particleDataWidth * particleDataHeight,
        channels: ['vx', 'vy', 'vz', 'w']
      }).toString(),
      outIdMassTintEdgePtr: readLinear({
        gl,
        texture: outIdMassTintEdgePtr,
        width: particleDataWidth, height: particleDataHeight,
        count: particleDataWidth * particleDataHeight,
        channels: ['id', 'mass', 'tint', 'edgePtr']
      }).toString()
    },
    {
      kSentinel: 'KPhysicsSentinel(4x4) #1',
      outParticlePositionAndSFC:
// all 1001/1002/1003/1004
`4x4 RGBA32F 16el
x      [1001.0                +1001.0] mean=1001.0 std=0 median=1001.0 |█               | nz=16/16 nearMin=16 nearMax=16
y      [1002.0                +1002.0] mean=1002.0 std=0 median=1002.0 |█               | nz=16/16 nearMin=16 nearMax=16
z      [1003.0                +1003.0] mean=1003.0 std=0 median=1003.0 |█               | nz=16/16 nearMin=16 nearMax=16
sfc    [1004.0                +1004.0] mean=1004.0 std=0 median=1004.0 |█               | nz=16/16 nearMin=16 nearMax=16`,
// all 2001/2002/2003/2004
      outParticleVelocity: `4x4 RGBA32F 16el
vx     [2001.0                +2001.0] mean=2001.0 std=0 median=2001.0 |█               | nz=16/16 nearMin=16 nearMax=16
vy     [2002.0                +2002.0] mean=2002.0 std=0 median=2002.0 |█               | nz=16/16 nearMin=16 nearMax=16
vz     [2003.0                +2003.0] mean=2003.0 std=0 median=2003.0 |█               | nz=16/16 nearMin=16 nearMax=16
w      [2004.0                +2004.0] mean=2004.0 std=0 median=2004.0 |█               | nz=16/16 nearMin=16 nearMax=16`,
// all 3001/3002/3003/4001
      outIdMassTintEdgePtr: `4x4 RGBA32F 16el
id     [3001.0                +3001.0] mean=3001.0 std=0 median=3001.0 |█               | nz=16/16 nearMin=16 nearMax=16
mass   [3002.0                +3002.0] mean=3002.0 std=0 median=3002.0 |█               | nz=16/16 nearMin=16 nearMax=16
tint   [3003.0                +3003.0] mean=3003.0 std=0 median=3003.0 |█               | nz=16/16 nearMin=16 nearMax=16
edgePtr[4001.0                +4001.0] mean=4001.0 std=0 median=4001.0 |█               | nz=16/16 nearMin=16 nearMax=16`
    }
  );

  kSentinel.dispose();
  // Textures are removed by layout.dispose(); avoid double-delete
  resetGL();
});

async function getGL() {
  const gl = await webGL2({ debug: true });
  return gl;
}

function resetGL() {
}