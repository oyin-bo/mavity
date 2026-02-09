// @ts-check

import { KEdgeCoarseMap } from './k-edge-coarse-map.js';
import { KEdgePrefixSum } from './k-edge-prefix-sum.js';
import { KEdgeRelocation } from './k-edge-relocation.js';
import { KIdentityMirror } from './k-identity-mirror.js';
import { KParticleReshuffle } from './k-particle-reshuffle.js';
import { KPhysicsSentinel } from './k-physics-sentinel.js';
import { KPhysicsSimulation } from './k-physics-simulation.js';
import { KSortEncoder } from './k-sort-encoder.js';
import { createTextureR32F, createTextureRGBA32F, createTextureRGBA32UI, glValidate } from './utils.js';
import { readLinear, formatNumber } from '../gravity/diag.js';
3
export class GraphLayout {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   edgeCount: number,
   *   texPosition: WebGLTexture,
   *   texVelocity: WebGLTexture,
   *   texIdMassTint: WebGLTexture,
   *   texEdgePtr: WebGLTexture,
   *   texEdgeStore: WebGLTexture,
   *   edgeCoarseMapStride?: number,
   *   gravityWindow?: number,
   *   dt?: number,
   *   G?: number,
   *   springK?: number,
   *   eps?: number,
   *   sfcResolution?: number
   * }} options
   */
  constructor({
    gl,
    particleCount,
    edgeCount,
    texPosition,
    texVelocity,
    texIdMassTint,
    texEdgePtr,
    texEdgeStore,
    edgeCoarseMapStride,
    gravityWindow,
    dt,
    G,
    springK,
    eps,
    sfcResolution
  }) {
    this.gl = gl;
    // Enable Mandatory Extension for rendering to FLOAT textures
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext)
      throw new Error('EXT_color_buffer_float not supported. GPGPU writes may fail.');

    if (!texPosition || !texVelocity || !texIdMassTint || !texEdgePtr || !texEdgeStore) {
      throw new Error('GraphLayout: Missing required textures in constructor');
    }

    this.particleCount = particleCount;
    this.edgeCount = edgeCount;

    this.edgeCoarseMapStride = edgeCoarseMapStride || 128;
    this.gravityWindow = gravityWindow || 16;
    this.dt = dt || 0.016;
    // Keep old layout behavior: default G is negative small (maintains prior repulsive convention),
    // and apply a damping that reproduces previous implicit kernel damping (0.998 => damping=0.002)
    this.G = G !== undefined ? G : -0.0001;
    this.springK = springK !== undefined ? springK : 1.0;
    this.eps = eps !== undefined ? eps : 0.1;
    this.damping = 0.002; // reproduces previous implicit damp of ~0.998
    this.sfcResolution = sfcResolution || 64.0;

    // Derived constants
    this.sortSpanSize = 128;

    this.encodedSortOrderWidth = Math.pow(2, Math.ceil(Math.log2(Math.sqrt(Math.ceil(particleCount / this.sortSpanSize)))));
    this.encodedSortOrderHeight = Math.ceil((particleCount / this.sortSpanSize) / this.encodedSortOrderWidth);

    this.edgeStoreWidth = Math.pow(2, Math.ceil(Math.log2(Math.sqrt(edgeCount))));
    this.edgeStoreHeight = Math.ceil(edgeCount / this.edgeStoreWidth);

    this.coarseWidth = Math.pow(2, Math.ceil(Math.log2(Math.sqrt(Math.ceil(edgeCount / this.edgeCoarseMapStride)))));
    this.coarseHeight = Math.ceil(Math.ceil(edgeCount / this.edgeCoarseMapStride) / this.coarseWidth);

    this.particleDataWidth = Math.pow(2, Math.ceil(Math.log2(Math.sqrt(this.particleCount + 1))));
    this.particleDataHeight = Math.ceil((this.particleCount + 1) / this.particleDataWidth);

    this.currentIdx = 0;
    this.passCounter = 0;
    this.renderCount = 0;

    // 1. Particle State Textures
    this.texPositionAndSFC = texPosition;
    this.texScratchPositionAndSFC = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });

    this.texVelocity = texVelocity;
    this.texScratchVelocity = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texIdMassTint = texIdMassTint;
    this.texScratchIdMassTint = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });

    // 2. Textures
    this.texEncodedSortOrder = [];
    for (let i = 0; i < 8; i++) {
      this.texEncodedSortOrder.push(createTextureRGBA32UI({ gl: this.gl, width: this.encodedSortOrderWidth, height: this.encodedSortOrderHeight }));
    }

    this.texEdgePtr = texEdgePtr;
    this.texScratchEdgePtr = createTextureR32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texEdgeStore = texEdgeStore;
    this.texScratchEdgeStore = createTextureR32F({ gl: this.gl, width: this.edgeStoreWidth, height: this.edgeStoreHeight });

    this.texCoarseMap = createTextureR32F({ gl: this.gl, width: this.coarseWidth, height: this.coarseHeight });

    // TODO: do NOT create these resources in the orchestrator ONLY in the respective kernels
    this.bufQuad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufQuad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.vaoQuad = gl.createVertexArray();
    gl.bindVertexArray(this.vaoQuad);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Sentinel kernel (writes distinctive values to the MRT targets so we can validate writes)
    this.kSentinel = new KPhysicsSentinel({
      gl: this.gl,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight
    });

    this.kPhysics = new KPhysicsSimulation({
      gl: this.gl,
      particleCount: this.particleCount,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      edgeStoreWidth: this.edgeStoreWidth,
      edgeStoreHeight: this.edgeStoreHeight,
      gravityWindow: this.gravityWindow,
      sfcResolution: this.sfcResolution,
      G: this.G,
      springK: this.springK,
      eps: this.eps,
      damping: this.damping
    });

    this.kSort = new KSortEncoder({
      gl: this.gl,
      particleCount: this.particleCount,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      sortSpanSize: this.sortSpanSize,
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      encodedSortOrderHeight: this.encodedSortOrderHeight
    });

    this.kParticleReshuffle = new KParticleReshuffle({
      gl: this.gl,
      particleCount: this.particleCount,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      sortSpanSize: this.sortSpanSize
    });

    this.kEdgePrefixSum = new KEdgePrefixSum({
      gl: this.gl,
      width: this.particleDataWidth,
      height: this.particleDataHeight,
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      particleCount: this.particleCount
    });

    this.kEdgeCoarseMap = new KEdgeCoarseMap({
      gl: this.gl,
      coarseWidth: this.coarseWidth,
      coarseHeight: this.coarseHeight,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      edgeCoarseMapStride: this.edgeCoarseMapStride,
      particleCount: this.particleCount
    });

    this.kEdgeRelocation = new KEdgeRelocation({
      gl: this.gl,
      edgeStoreWidth: this.edgeStoreWidth,
      edgeStoreHeight: this.edgeStoreHeight,
      coarseWidth: this.coarseWidth,
      coarseHeight: this.coarseHeight,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      startStride: this.edgeCoarseMapStride,
      edgeCount: this.edgeCount,
      particleCount: this.particleCount
    });

    this.kIdentity = new KIdentityMirror({
      gl: this.gl,
      particleCount: this.particleCount,
      textureWidth: this.particleDataWidth, // The identity map (output) is PID-based
      textureHeight: this.particleDataHeight
    });
    
    this.frameCounter = 0;
    this._lastDebugTime = Date.now() - 18000; // Force first debug after 2s

    this._debugFbo = gl.createFramebuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  run() {
    // Clear any previous errors
    let err = this.gl.getError();
    while (err !== this.gl.NO_ERROR) {
      console.warn('Ignoring Pre-Run WebGL Error:', err);
      err = this.gl.getError();
    }
    glValidate(this.gl, 'RelocationSystem Run Entry');
    // Isolate GPGPU State from Three.js
    const prevBlend = this.gl.isEnabled(this.gl.BLEND);
    const prevDepth = this.gl.isEnabled(this.gl.DEPTH_TEST);
    const prevScissor = this.gl.isEnabled(this.gl.SCISSOR_TEST);
    const prevStencil = this.gl.isEnabled(this.gl.STENCIL_TEST);

    this.gl.disable(this.gl.BLEND);
    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.SCISSOR_TEST);
    this.gl.disable(this.gl.STENCIL_TEST);
    this.gl.colorMask(true, true, true, true);

    const sortOffset = (this.passCounter % 2) * (this.sortSpanSize / 2);

    const now = Date.now();
    const doDebug = (now - this._lastDebugTime >= 20000) && typeof window !== 'undefined' && /** @type {*} */(window).LOG_NEXT_FRAME;
    if (typeof window !== 'undefined' && /** @type {*} */(window).LOG_NEXT_FRAME) {
      /** @type {*} */(window).LOG_NEXT_FRAME = false;
    }

    const before = null;

    // Sentinel: write sentinel values into the NEXT buffer so we can validate the pipeline
    this.kSentinel.run({
      outParticlePositionAndSFC: this.texScratchPositionAndSFC,
      outParticleVelocity: this.texScratchVelocity,
      outIdMassTintEdgePtr: this.texScratchIdMassTint
    });

    const afterSentinel = null;

    // 2. Physics: Integrate (Read current, Write next)
    this.kPhysics.run({
      inParticlePositionAndSFC: this.texPositionAndSFC,
      inParticleVelocity: this.texVelocity,
      inIdMassTintEdgePtr: this.texIdMassTint,
      inTexEdgePtr: this.texEdgePtr,
      inTexEdgeStore: this.texEdgeStore,

      outParticlePositionAndSFC: this.texScratchPositionAndSFC,
      outParticleVelocity: this.texScratchVelocity,
      outIdMassTintEdgePtr: this.texScratchIdMassTint,

      dt: this.dt
    });

    const afterPhysics = null;

    // 2. Sort: Generate Atlas (Read next which has NEW SFC in pos.w)
    this.kSort.run({
      texParticlePositionAndSFC: this.texScratchPositionAndSFC,
      outTexEncodedOrder: this.texEncodedSortOrder,
      sortOffset
    });

    // 3. Reshuffle: Apply Sort (Read next, Write current)
    // Recycles 'current' buffer to hold the Sorted state.
    this.kParticleReshuffle.run({
      inParticlePositionAndSFC: this.texScratchPositionAndSFC,
      inParticleVelocity: this.texScratchVelocity,
      inIdMassTintEdgePtr: this.texScratchIdMassTint,
      inEncodedSortOrder: this.texEncodedSortOrder,

      outParticlePositionAndSFC: this.texPositionAndSFC,
      outParticleVelocity: this.texVelocity,
      outIdMassTintEdgePtr: this.texIdMassTint,

      sortOffset
    });

    const afterReshuffle = null;

    // 5. Identity: Update Map (Read current, Write Static Identity)
    // MOVED: Must run BEFORE relocation so KReloc has access to PID->NewPhys
    this.kIdentity.run({
      texIdMassTintEdgePtr: this.texIdMassTint
    });

    // 4. Edge Relocation Pipeline
    // A. Prefix Sum (IdMassTintEdgePtr[current] -> EdgePtr[next])
    // Note: 'current' has the reshuffled particles
    this.kEdgePrefixSum.run({
      inIdMassTintEdgePtr: this.texIdMassTint,
      texEdgePtrOld: this.texEdgePtr, // Old Layout Ptrs (before prefix sum)
      texSortOrder: this.texEncodedSortOrder,
      texOutput: this.texScratchEdgePtr, // Writing to Next to avoid overwriting Current (Old)
      sortOffset, // Pass current sort offset
      quadVAO: this.vaoQuad
    });

    const afterPrefixSum = null;

    // B. Coarse Map (EdgePtr[next] -> CoarseMap)
    this.kEdgeCoarseMap.run({
      texEdgePtr: this.texScratchEdgePtr,
      texCoarseMap: this.texCoarseMap
    });

    const debugCoarseMap = null;

    // C. Relocation (Store[old] -> Store[new])

    this.kEdgeRelocation.run({
      texEdgeStoreOld: this.texEdgeStore,
      texEdgeStoreNew: this.texScratchEdgeStore,
      texCoarseMap: this.texCoarseMap,
      texEdgePtrNew: this.texScratchEdgePtr,
      texEdgePtrOld: this.texEdgePtr, // Old Layout Ptrs
      texIdMassTintEdgePtrOld: this.texScratchIdMassTint,
      texIdentityNew: this.kIdentity.texIdentity,
      slotsRes: { x: this.particleDataWidth, y: this.particleDataHeight },
      texSortOrder: this.texEncodedSortOrder,
      sortOffset // Pass into Relocation
    });

    [this.texEdgeStore, this.texScratchEdgeStore] = [this.texScratchEdgeStore, this.texEdgeStore];
    [this.texEdgePtr, this.texScratchEdgePtr] = [this.texScratchEdgePtr, this.texEdgePtr];

    this.passCounter++;
    this.frameCounter++;
    this.renderCount++;

    const detailedEdgeTrace = null;

    if (doDebug) {
      console.log('Frame Update Complete');
      this._lastDebugTime = now;
    }

    // Restore State
    if (prevBlend) this.gl.enable(this.gl.BLEND);
    if (prevDepth) this.gl.enable(this.gl.DEPTH_TEST);
    if (prevScissor) this.gl.enable(this.gl.SCISSOR_TEST);
    if (prevStencil) this.gl.enable(this.gl.STENCIL_TEST);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  /**
   * @param {{ pixels?: boolean }} [options]
   */
  valueOf({ pixels } = {}) {
    return {
      particleCount: this.particleCount,
      edgeCount: this.edgeCount,
      particleDataWidth: this.particleDataWidth,
      particleDataHeight: this.particleDataHeight,
      edgeStoreWidth: this.edgeStoreWidth,
      edgeStoreHeight: this.edgeStoreHeight,
      coarseWidth: this.coarseWidth,
      coarseHeight: this.coarseHeight,
      renderCount: this.renderCount,
      frameCounter: this.frameCounter,
      passCounter: this.passCounter,
      dt: this.dt,
      G: this.G,
      springK: this.springK,
      eps: this.eps,
      gravityWindow: this.gravityWindow,
      sfcResolution: this.sfcResolution,
      
      position: readLinear({
        gl: this.gl,
        texture: this.texPositionAndSFC,
        width: this.particleDataWidth,
        height: this.particleDataHeight,
        count: this.particleCount,
        channels: ['x', 'y', 'z', 'sfc'],
        pixels
      }),
      velocity: readLinear({
        gl: this.gl,
        texture: this.texVelocity,
        width: this.particleDataWidth,
        height: this.particleDataHeight,
        count: this.particleCount,
        channels: ['vx', 'vy', 'vz', 'w'],
        pixels
      }),
      idMassTint: readLinear({
        gl: this.gl,
        texture: this.texIdMassTint,
        width: this.particleDataWidth,
        height: this.particleDataHeight,
        count: this.particleCount,
        channels: ['pid', 'mass', 'tint', 'unused'],
        pixels
      }),
      edgePtr: readLinear({
        gl: this.gl,
        texture: this.texEdgePtr,
        width: this.particleDataWidth,
        height: this.particleDataHeight,
        count: this.particleCount + 1,
        channels: ['offset'],
        format: 'R32F',
        pixels
      }),
      edgeStore: readLinear({
        gl: this.gl,
        texture: this.texEdgeStore,
        width: this.edgeStoreWidth,
        height: this.edgeStoreHeight,
        count: this.edgeCount,
        channels: ['targetIdx'],
        format: 'R32F',
        pixels
      }),

      kPhysics: this.kPhysics.valueOf?.({ pixels }),
      kSort: this.kSort.valueOf?.({ pixels }),
      kParticleReshuffle: this.kParticleReshuffle.valueOf?.({ pixels }),
      kEdgePrefixSum: this.kEdgePrefixSum.valueOf?.({ pixels }),
      kEdgeCoarseMap: this.kEdgeCoarseMap.valueOf?.({ pixels }),
      kEdgeRelocation: this.kEdgeRelocation.valueOf?.({ pixels }),
      kIdentity: this.kIdentity.valueOf?.({ pixels }),

      toString: () => this.toString()
    };
  }

  toString() {
    const val = this.valueOf({ pixels: false });
    return `GraphLayout(${this.particleCount} particles, ${this.edgeCount} edges) #${this.renderCount}
  dt=${formatNumber(this.dt)} G=${formatNumber(this.G)} springK=${formatNumber(this.springK)}
  pos: ${val.position}
  vel: ${val.velocity}
  idMass: ${val.idMassTint}
  edgePtr: ${val.edgePtr}
  edgeStore: ${val.edgeStore}`;
  }

  dispose() {
    // Dispose buffers
    this.gl.deleteBuffer(this.bufQuad);

    // Dispose textures
    this.texEncodedSortOrder.forEach(t => this.gl.deleteTexture(t));
    this.gl.deleteTexture(this.texEdgePtr);
    this.gl.deleteTexture(this.texScratchEdgePtr);
    this.gl.deleteTexture(this.texPositionAndSFC);
    this.gl.deleteTexture(this.texScratchPositionAndSFC);
    this.gl.deleteTexture(this.texVelocity);
    this.gl.deleteTexture(this.texScratchVelocity);
    this.gl.deleteTexture(this.texIdMassTint);
    this.gl.deleteTexture(this.texScratchIdMassTint);
    this.gl.deleteTexture(this.texEdgeStore);
    this.gl.deleteTexture(this.texScratchEdgeStore);
    this.gl.deleteTexture(this.texCoarseMap);

    // Dispose framebuffers
    this.gl.deleteFramebuffer(this._debugFbo);

    // Dispose vertex arrays
    this.gl.deleteVertexArray(this.vaoQuad);

    // Dispose child systems
    this.kSentinel.dispose?.();
    this.kPhysics.dispose?.();
    this.kSort.dispose?.();
    this.kParticleReshuffle.dispose?.();
    this.kEdgePrefixSum.dispose?.();
    this.kEdgeCoarseMap.dispose?.();
    this.kEdgeRelocation.dispose?.();
    this.kIdentity.dispose();
  }
}
