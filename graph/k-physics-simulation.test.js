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
      particleDataWidth: kSentinel.particleDataWidth,
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
    particleCount: 3,
    particleDataWidth: 4,
    particleDataHeight: 1,
    G: -1.0, // negative to preserve previous repulsive behaviour
    damping: 0.002, // reproduce previous implicit damping (0.998)
    eps: 0.0001,
    particles: [
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 }, // P0
      { x: 1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 }, // P1 - will repel P0
      { x: 0, y: 1, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 } // P2 - connected by edge to P0
    ]
  });

  engine.updateEdges(
    [0, 1, 1, 1], // P0 start at 0, P1 start at 1...
    [2] // P0 -> P2
  );

  const { positions, velocities } = engine.run({ dt: 0.1 });

  // Math check for P0:
  // p = (0,0,0), v = (0,0,0)
  // Gravity from P1 (at 1,0,0): dp=(1,0,0), dist=1, acc_g1 = -1 * (1,0,0) = (-1, 0, 0)
  // Gravity from P2 (at 0,1,0): dp=(0,1,0), dist=1, acc_g2 = -1 * (0,1,0) = (0, -1, 0)
  // Edge to P2: dp=(0,1,0), acc_e = 1 * (0,1,0) = (0, 1, 0)
  // Total acc = (-1, 0, 0)  <-- Y components cancel out!
  // v_next = (0 + (-1, 0, 0)*0.1) * 0.998 = (-0.0998, 0, 0)
  // p_next = (0,0,0) + (-0.0998, 0, 0)*0.1 = (-0.00998, 0, 0)

  assert.ok(Math.abs(velocities[0].vx - (-0.0998)) < 0.0001, `P0 Vel X mismatch: ${velocities[0].vx}`);
  assert.ok(Math.abs(velocities[0].vy - (0.0)) < 0.0001, `P0 Vel Y mismatch: ${velocities[0].vy}`);
  assert.ok(Math.abs(positions[0].x - (-0.00998)) < 0.0001, `P0 Pos X mismatch: ${positions[0].x}`);
  assert.ok(Math.abs(positions[0].y - (0.0)) < 0.0001, `P0 Pos Y mismatch: ${positions[0].y}`);

  // P1: (1,0,0). Feelings force from P0 (0,0,0) and P2 (0,1,0)
  // Grav from P0: dp = (-1,0,0), acc = -1 * (-1,0,0) = (1, 0, 0)
  // P1 should be moving POSITIVE X
  assert.ok(velocities[1].vx > 0, `P1 should move positive X (repelled by P0), but was ${velocities[1].vx}`);
});

test('KPhysicsSimulation: Solar System Orbit (Sun-Earth SI)', async () => {
  // G ~= 6.6743e-11
  // Sun: mass ~= 1.989e30
  // Earth: mass ~= 5.972e24, dist ~= 1.496e11, v ~= 29782
  const G = 6.6743e-11;
  const sunMass = 1.989e30;
  const earthMass = 5.972e24;
  const earthDist = 1.496e11;
  const earthVel = 29782; 
  const dt = 60; // 60s step for stability
  const iterations = 3000; 

  const engine = await createEngine({
    particleCount: 2,
    particleDataWidth: 4, 
    particleDataHeight: 1,
    G,
    damping: 0,
    eps: 1000.0, 
    particles: [
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: sunMass }, // Sun
      { x: earthDist, y: 0, z: 0, vx: 0, vy: earthVel, vz: 0, mass: earthMass } // Earth
    ]
  });

  // Verify initial state read-back (Float32 precision at 1e11 is ~1.6e4)
  const initial = engine.readParticles();
  assert.ok(Math.abs(initial.positions[1].x - earthDist) < 20000, `Initial Earth X mismatch: ${initial.positions[1].x}`);
  assert.ok(Math.abs(initial.velocities[1].vy - earthVel) < 1.0, `Initial Earth VY mismatch: ${initial.velocities[1].vy}`);

  const startDist = earthDist;
  
  // Run 3000 iterations (uses internal ping-pong)
  const startTime = Date.now();
  const { positions, velocities } = engine.run({ dt, steps: iterations });
  const endTime = Date.now();

  const sun = positions[0];
  const earth = positions[1];
  const ev = velocities[1];

  const finalDist = Math.sqrt(earth.x * earth.x + earth.y * earth.y + earth.z * earth.z);
  const distErrorPercent = Math.abs(finalDist - startDist) / startDist * 100;

  // Verify distance stability (within 0.1% for 3000h)
  assert.ok(distErrorPercent < 0.1, `Orbit unstable: distance error ${distErrorPercent.toFixed(4)}% > 0.1%`);
  
  // Verify angle (should have moved ~180,000s / ~2 days)
  const angle = Math.atan2(earth.y, earth.x);
  assert.ok(Math.abs(angle) > 0.03, `Earth didn't move enough: ${angle.toFixed(4)} rad`);

  engine.dispose();
});

test('KPhysicsSimulation: Full Solar Year Orbit (60k iterations)', async () => {
  const G = 6.6743e-11;
  const sunMass = 1.989e30;
  const earthMass = 5.972e24;
  const earthDist = 1.496e11;
  const earthVel = 29782; 
  
  // We want more iterations for better stability over a full year
  const iterations = 60000;
  const totalTime = 31558149; // More precise orbital period for r=1.496e11
  const dt = totalTime / iterations; 

  const engine = await createEngine({
    particleCount: 2,
    particleDataWidth: 4, 
    particleDataHeight: 1,
    G,
    damping: 0,
    eps: 1000.0,
    particles: [
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: sunMass }, // Sun
      { x: earthDist, y: 0, z: 0, vx: 0, vy: earthVel, vz: 0, mass: earthMass } // Earth
    ]
  });

  const startX = earthDist;
  
  console.log(`Starting Full Solar Year simulation: ${iterations} steps, dt=${dt.toFixed(2)}s`);
  const startTime = Date.now();
  
  // Run in chunks
  const chunks = 10;
  let finalRes;
  for (let i = 0; i < chunks; i++) {
    finalRes = engine.run({ dt, steps: iterations / chunks });
  }
  
  const endTime = Date.now();
  const earth = finalRes.positions[1];
  const finalDist = Math.sqrt(earth.x * earth.x + earth.y * earth.y + earth.z * earth.z);
  const distErrorPercent = Math.abs(finalDist - earthDist) / earthDist * 100;
  const angle = Math.atan2(earth.y, earth.x);

  console.log(`Full Year complete in ${endTime - startTime}ms`);
  console.log(`Distance Error: ${distErrorPercent.toFixed(4)}%`);
  console.log(`Final Angle: ${angle.toFixed(4)} rad`);

  // Assert Earth is approximately back at the same spot.
  assert.ok(distErrorPercent < 0.1, `Orbit drifted too far: ${distErrorPercent.toFixed(4)}%`);
  assert.ok(Math.abs(angle) < 0.02, `Earth not back at start angle: ${angle.toFixed(4)} rad`);

  engine.dispose();
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
 *  eps?: number,
 *  damping?: number
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
  const damping = options.damping !== undefined ? options.damping : 0.0;

  // Mock textures
  let inTexPos = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  let inTexVel = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  let inTexIdMass = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });

  const inTexEdgePtr = createTextureR32F({ gl, width: particleDataWidth, height: particleDataHeight });
  const inTexEdgeStore = createTextureR32F({ gl, width: edgeStoreWidth, height: edgeStoreHeight });

  let outTexPos = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  let outTexVel = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });
  let outTexIdMass = createTextureRGBA32F({ gl, width: particleDataWidth, height: particleDataHeight });

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
    eps,
    damping
  });

  if (options.particles) updateParticles(options.particles);

  return {
    run,
    updateParticles,
    updateEdges,
    readParticles,
    dispose
  };

  /**
   * @param {{
   *  x: number, y: number, z: number,
   *  vx: number, vy: number, vz: number,
   *  mass: number
   * }[]} particles 
   */
  function updateParticles(particles) {
    const posData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    const velData = new Float32Array(particleDataWidth * particleDataHeight * 4);
    const idMassData = new Float32Array(particleDataWidth * particleDataHeight * 4);

    particles.forEach((p, i) => {
      posData.set([p.x, p.y, p.z, 0], i * 4);
      velData.set([p.vx, p.vy, p.vz, 0], i * 4);
      idMassData.set([i, p.mass, 0, 0], i * 4);
    });

    writeTextureRGBA32F({ gl, texture: inTexPos, width: particleDataWidth, height: particleDataHeight, data: posData });
    writeTextureRGBA32F({ gl, texture: inTexVel, width: particleDataWidth, height: particleDataHeight, data: velData });
    writeTextureRGBA32F({ gl, texture: inTexIdMass, width: particleDataWidth, height: particleDataHeight, data: idMassData });
  }

  /**
   * @param {number[]} ptrs 
   * @param {number[]} store 
   */
  function updateEdges(ptrs, store) {
    const edgePtrData = new Float32Array(particleDataWidth * particleDataHeight);
    edgePtrData.set(ptrs);
    writeTextureR32F({ gl, texture: inTexEdgePtr, width: particleDataWidth, height: particleDataHeight, data: edgePtrData });

    const edgeStoreData = new Float32Array(edgeStoreWidth * edgeStoreHeight);
    edgeStoreData.fill(-1);
    edgeStoreData.set(store);
    writeTextureR32F({ gl, texture: inTexEdgeStore, width: edgeStoreWidth, height: edgeStoreHeight, data: edgeStoreData });
  }

  function readParticles() {
    // Read back results
    const resPos = readLinear({
      gl,
      texture: inTexPos,
      width: particleDataWidth,
      height: particleDataHeight,
      channels: ['x', 'y', 'z', 'sfc'],
      pixels: true
    });

    const resVel = readLinear({
      gl,
      texture: inTexVel,
      width: particleDataWidth,
      height: particleDataHeight,
      channels: ['vx', 'vy', 'vz', 'w'],
      pixels: true
    });

    const resIdMass = readLinear({
      gl,
      texture: inTexIdMass,
      width: particleDataWidth,
      height: particleDataHeight,
      channels: ['id', 'mass', 'tint', 'ptr'],
      pixels: true
    });

    return {
      positions: /** @type {{ x: number, y: number, z: number, sfc: number }[]} */ (resPos.pixels),
      velocities: /** @type {{ vx: number, vy: number, vz: number, w: number }[]} */ (resVel.pixels),
      idMass: /** @type {{ id: number, mass: number, tint: number, ptr: number }[]} */ (resIdMass.pixels)
    };
  }

  /**
   * @param {{
   *  dt: number,
   *  steps?: number
   * }} options
   */
  function run({ dt, steps = 1 }) {

    for (let i = 0; i < steps; i++) {
      kSentinel.run({
        outParticlePositionAndSFC: outTexPos,
        outParticleVelocity: outTexVel,
        outIdMassTintEdgePtr: outTexIdMass,
        inParticlePositionAndSFC: inTexPos,
        inParticleVelocity: inTexVel,
        inIdMassTintEdgePtr: inTexIdMass,
        inTexEdgePtr,
        inTexEdgeStore,
        dt
      });

      // Swap
      [inTexPos, outTexPos] = [outTexPos, inTexPos];
      [inTexVel, outTexVel] = [outTexVel, inTexVel];
      [inTexIdMass, outTexIdMass] = [outTexIdMass, inTexIdMass];
    }

    return readParticles();
  }

  function dispose() {
    kSentinel.dispose();
    resetGL();
  }
}

async function getGL() {
  /** @type {WebGL2RenderingContext} */
  const gl = await webGL2({ debug: true, width: 1024, height: 1024 });
  return gl;
}

function resetGL() {
}