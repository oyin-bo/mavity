// @ts-check

import assert from 'node:assert';
import { test } from 'node:test';


import { KPhysicsSimulation } from './k-physics-simulation.js';

import { webGL2 } from 'webgl2';
import { createTextureR32F, createTextureRGBA32F, writeTextureRGBA32F, writeTextureR32F } from './utils.js';
import { readLinear } from '../gravity/diag.js';

test('KPhysicsSentinel: basic allocation', async () => {
  const gl = await getGL();

  const particleCount = 10;
  const particleDataWidth = 4;
  const particleDataHeight = 4;
  const edgeCount = 20;
  const edgeStoreWidth = 5;
  const edgeStoreHeight = 4;
  const sfcResolution = 8;
  const gravityWindow = 4;

  const kSentinel = new KPhysicsSimulation({
    gl,
    particleCount,
    particleDataWidth,
    particleDataHeight,
    edgeStoreWidth,
    edgeStoreHeight,
    sfcResolution,
    gravityWindow
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

test('KPhysicsSimulation: NEW gravity and edges', async () => {

  const engine = await createEngine({
    particles: [
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 }, // P0
      { x: 1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 }, // P1 - will repel P0
      { x: 0, y: 1, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 } // P2 - connected by edge to P0
    ]
  });

  const { positions, velocities } = engine.run({ dt: 0.1 });

  // Math check for P0:
  // p = (0,0,0), v = (0,0,0)
  // Gravity from P1: dp=(1,0,0), acc_g = -1 * (1,0,0) = (-1, 0, 0)
  // Edge to P2: dp=(0,1,0), acc_e = 1 * (0,1,0) = (0, 1, 0)
  // Total acc = (-1, 1, 0)
  // v_next = (0 + (-1, 1, 0)*0.1) * 0.998 = (-0.0998, 0.0998, 0)
  // p_next = (0,0,0) + (-0.0998, 0.0998, 0)*0.1 = (-0.00998, 0.00998, 0)

  assert.ok(Math.abs(velocities[0].vx - (-0.0998)) < 0.0001, `P0 Vel X mismatch: ${velocities[0].vx}`);
  assert.ok(Math.abs(velocities[0].vy - (0.0998)) < 0.0001, `P0 Vel Y mismatch: ${velocities[0].vy}`);
  assert.ok(Math.abs(positions[0].x - (-0.00998)) < 0.0001, `P0 Pos X mismatch: ${positions[0].x}`);
  assert.ok(Math.abs(positions[0].y - (0.00998)) < 0.0001, `P0 Pos Y mismatch: ${positions[0].y}`);

});

/**
 * Prepare a simulation engine to run scenarios.
 * @param {{
 *  particles: {
 *    x: number, y: number, z: number,
 *    vx: number, vy: number, vz: number,
 *    mass: number
 *  }[],
 *  gl?: WebGL2RenderingContext,
 *  particleCount?: number,
 *  particleDataWidth?: number,
 *  particleDataHeight?: number,
 *  edgeStoreWidth?: number,
 *  edgeStoreHeight?: number,
 *  sfcResolution?: number,
 *  gravityWindow?: number,
 *  G?: number,
 *  springK?: number,
 *  eps?: number
 * }} options
 */
async function createEngine(options) {
  let ownGL = false;
  const gl = options.gl || ((ownGL = true) && await getGL());
  const particleCount = options.particleCount || 10;
  const particleDataWidth = options.particleDataWidth || 4;
  const particleDataHeight = options.particleDataHeight || 4;
  const edgeStoreWidth = options.edgeStoreWidth || 5;
  const edgeStoreHeight = options.edgeStoreHeight || 4;
  const sfcResolution = options.sfcResolution || 8;
  const gravityWindow = options.gravityWindow || 4;
  const G = options.G || 1.0;
  const springK = options.springK || 1.0;
  const eps = options.eps || 0.0;

  // Mock textures
  const inParticlePositionAndSFC = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const inParticleVelocity = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const inIdMassTintEdgePtr = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const inTexEdgePtr = createTextureR32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const inTexEdgeStore = createTextureR32F({ gl, width: edgeStoreWidth, height: edgeStoreHeight });

  const outParticlePositionAndSFC = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outParticleVelocity = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const outIdMassTintEdgePtr = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });

  const kSentinel = new KPhysicsSimulation({
    gl,
    particleCount,
    particleDataWidth,
    particleDataHeight,
    edgeStoreWidth,
    edgeStoreHeight,
    sfcResolution,
    gravityWindow,
    G,
    springK,
    eps
  });

  updateParticles(options.particles);

  return {
    run,
    updateParticles,
    readParticles,
    dispose
  };

  /**
   * @param {{
   *  x: number, y: number, z: number,
   *  vx: number, vy: number, vz: number,
   *  mass: number
   * }[]} positions 
   */
  function updateParticles(positions) {
    const posData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    const velData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    const idMassData = new Float32Array(particleDataWidth * particleDataHeight * 4);

    positions.forEach((p, i) => {
      posData.set([p.x, p.y, p.z, 0], i * 4);
      velData.set([p.vx, p.vy, p.vz, 0], i * 4);
      idMassData.set([i, p.mass, 0, 0], i * 4);
    });

    writeTextureRGBA32F({ gl, texture: inParticlePositionAndSFC, width: particleDataWidth, height: particleDataHeight, data: posData });
    writeTextureRGBA32F({ gl, texture: inParticleVelocity, width: particleDataWidth, height: particleDataHeight, data: velData });
    writeTextureRGBA32F({ gl, texture: inIdMassTintEdgePtr, width: particleDataWidth, height: particleDataHeight, data: idMassData });
  }

  function readParticles() {
    // Read back results
    const resPos = readLinear({
      gl,
      texture: outParticlePositionAndSFC,
      width: particleDataWidth,
      height: particleDataHeight,
      channels: ['x', 'y', 'z', 'sfc'],
      pixels: true
    });

    const resVel = readLinear({
      gl,
      texture: outParticleVelocity,
      width: particleDataWidth,
      height: particleDataHeight,
      channels: ['vx', 'vy', 'vz', 'w'],
      pixels: true
    });

    return {
      positions: /** @type {{ x: number, y: number, z: number, sfc: number }[]} */ (resPos.pixels),
      velocities: /** @type {{ vx: number, vy: number, vz: number, w: number }[]} */ (resVel.pixels)
    };
  }

  /**
   * @param {{
   *  dt: number
   * }} options
   */
  function run({ dt }) {

    // Initialize data
    // P0 at (0,0,0)
    // P1 at (1,0,0) - will repel P0
    // P2 at (0,1,0) - connected by edge to P0
    const posData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    posData.set([0, 0, 0, 0], 0);
    posData.set([1, 0, 0, 0], 4);
    posData.set([0, 1, 0, 0], 8);
    writeTextureRGBA32F({ gl, texture: inParticlePositionAndSFC, width: particleDataWidth, height: particleDataHeight, data: posData });

    const velData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    writeTextureRGBA32F({ gl, texture: inParticleVelocity, width: particleDataWidth, height: particleDataHeight, data: velData });

    const idMassData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    writeTextureRGBA32F({ gl, texture: inIdMassTintEdgePtr, width: particleDataWidth, height: particleDataHeight, data: idMassData });

    // Edge pointers: P0 has 1 edge starting at 0, P1 at 1, P2 at 1, P3...
    const edgePtrData = new Float32Array(particleDataWidth * particleDataHeight);
    edgePtrData[0] = 0; // P0 edge start
    edgePtrData[1] = 1; // P1 edge start (so P0 end)
    edgePtrData[2] = 1; // P2 edge start (so P1 end)
    edgePtrData[3] = 1; // P3 edge start (so P2 end)
    writeTextureR32F({ gl, texture: inTexEdgePtr, width: particleDataWidth, height: particleDataHeight, data: edgePtrData });

    // Edge store: P0 -> P2
    const edgeStoreData = new Float32Array(edgeStoreWidth * edgeStoreHeight);
    edgeStoreData.fill(-1);
    edgeStoreData[0] = 2; // P0's first (and only) edge points to P2
    writeTextureR32F({ gl, texture: inTexEdgeStore, width: edgeStoreWidth, height: edgeStoreHeight, data: edgeStoreData });

    // Run one frame
    kSentinel.run({
      outParticlePositionAndSFC,
      outParticleVelocity,
      outIdMassTintEdgePtr,
      inParticlePositionAndSFC,
      inParticleVelocity,
      inIdMassTintEdgePtr,
      inTexEdgePtr,
      inTexEdgeStore,
      dt
    });

    return readParticles();
  }

  function dispose() {
    kSentinel.dispose();
    resetGL();
  }
}

async function getGL() {
  /** @type {WebGL2RenderingContext} */
  const gl = await webGL2({ debug: true });
  return gl;
}

function resetGL() {
}