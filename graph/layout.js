// @ts-check

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

export class GraphLayout {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   edgeCount: number,
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
      console.error('EXT_color_buffer_float not supported. GPGPU writes may fail.');

    this.particleCount = particleCount;
    this.edgeCount = edgeCount;

    this.edgeCoarseMapStride = edgeCoarseMapStride || 128;
    this.gravityWindow = gravityWindow || 16;
    this.dt = dt || 0.016;
    this.G = G || 0.0001;
    this.springK = springK || 1.0;
    this.eps = eps || 0.1;
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

    // 1. Particle State Textures (Ping-Pong)
    this.texPositionAndSFC = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texScratchPositionAndSFC = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });

    this.texVelocity = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texScratchVelocity = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texIdMassTint = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texScratchIdMassTint = createTextureRGBA32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });

    // 2. Textures
    this.texEncodedSortOrder = [];
    for (let i = 0; i < 8; i++) {
      this.texEncodedSortOrder.push(createTextureRGBA32UI({ gl: this.gl, width: this.encodedSortOrderWidth, height: this.encodedSortOrderHeight }));
    }

    this.texEdgePtr = createTextureR32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texScratchEdgePtr = createTextureR32F({ gl: this.gl, width: this.particleDataWidth, height: this.particleDataHeight });
    this.texEdgeStore = createTextureR32F({ gl: this.gl, width: this.edgeStoreWidth, height: this.edgeStoreHeight });
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
      eps: this.eps
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


    // CPU-side Ghost Map: PID -> [x, y, z]
    // We'll store it as a Float32Array where index = PID * 3 (or 4 to align?)
    // Plan says: "index is the PID and the value is the XYZ"
    // Let's use stride 4 for alignment/simplicity (XYZW)
    // Assuming max PID is closely related to particleCount, but PIDs are 30-bit?
    // "PID Mapping: The CPU maintains a fast lookup (potentially just cutting middle 30 bit of DID) to translate BlueSky DIDs into 32-bit Persistent IDs (PIDs)."
    // If PIDs are dense and 0-indexed up to particleCount, we can use array.
    // If PIDs are sparse 30-bit hashes, we need a Map or a very large array (4GB for 1B PIDs).
    // The plan says: "Float32Array where the index is the PID". This implies dense PIDs or a large array.
    // For now, let's assume PIDs are managed and mapped to 0..N range or we allocate for Max PID.
    // But wait, "2.5M daily active accounts".
    // If we use 30-bit hash as PID, the array size is 2^30 * 16 bytes = 16GB. That's too big.
    // So there must be a "PID -> Compact Index" mapping or PIDs are assigned sequentially.
    // "Translate BlueSky DIDs into 32-bit Persistent IDs (PIDs)."
    // Let's assume for now we use a fixed size buffer based on particleCount, assuming PIDs < particleCount * 2 roughly.
    // Or just start with particleCount * 4 size.
    this.cpuGhostMap = new Float32Array(this.particleCount * 4); 
    this.cpuEdgePtrMap = new Float32Array(this.particleCount); // PID -> EdgePtr
    this.cpuEdgeStore = new Float32Array(this.edgeCount); // Raw Edge Data
    
    this.frameCounter = 0;
    this._lastDebugTime = Date.now() - 18000; // Force first debug after 2s

    this._debugFbo = gl.createFramebuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  /**
   * @param {{
   *   particles: Float32Array,
   *   edgePtr: Uint32Array,
   *   edgeStore: Uint32Array
   * }} data
   */
  seed(data) {
    const gl = this.gl;
    glValidate(gl, 'Seed Start');

    // De-interleave data for textures
    const texelCount = this.particleDataWidth * this.particleDataHeight;
    const posData = new Float32Array(texelCount * 4);
    const velData = new Float32Array(texelCount * 4);
    const idMassTintEdgePtrData = new Float32Array(texelCount * 4);

    for (let i = 0; i < this.particleCount; i++) {
      const off = i * 12;
      posData[i * 4 + 0] = data.particles[off + 0];
      posData[i * 4 + 1] = data.particles[off + 1];
      posData[i * 4 + 2] = data.particles[off + 2];
      posData[i * 4 + 3] = 0.0; // empty w channel for position

      velData[i * 4 + 0] = data.particles[off + 4];
      velData[i * 4 + 1] = data.particles[off + 5];
      velData[i * 4 + 2] = data.particles[off + 6];
      velData[i * 4 + 3] = 0.0; // SFC Key (calculated in step)

      idMassTintEdgePtrData[i * 4 + 0] = data.particles[off + 7]; // PID (Persistent)
      idMassTintEdgePtrData[i * 4 + 1] = data.particles[off + 3]; // Mass (Moved from Pos.w)
      idMassTintEdgePtrData[i * 4 + 2] = 0.0; // Color/Tint
      // STORE POINTER (INDEX)
      idMassTintEdgePtrData[i * 4 + 3] = data.edgePtr[i];
    }

    // Upload to textures
    gl.bindTexture(gl.TEXTURE_2D, this.texPositionAndSFC);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.particleDataWidth, this.particleDataHeight, gl.RGBA, gl.FLOAT, posData);

    gl.bindTexture(gl.TEXTURE_2D, this.texVelocity);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.particleDataWidth, this.particleDataHeight, gl.RGBA, gl.FLOAT, velData);

    gl.bindTexture(gl.TEXTURE_2D, this.texIdMassTint);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.particleDataWidth, this.particleDataHeight, gl.RGBA, gl.FLOAT, idMassTintEdgePtrData);

    gl.bindTexture(gl.TEXTURE_2D, this.texEdgePtr);
    // Upload N+1 entries to include Sentinel
    const ptrTotal = this.particleCount + 1;
    const ptrRows = Math.floor(ptrTotal / this.particleDataWidth);
    const ptrRem = ptrTotal % this.particleDataWidth;
    const edgePtrFloat = new Float32Array(data.edgePtr);
    if (ptrRows > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.particleDataWidth, ptrRows, gl.RED, gl.FLOAT, edgePtrFloat);
    }
    if (ptrRem > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, ptrRows, ptrRem, 1, gl.RED, gl.FLOAT, edgePtrFloat, ptrRows * this.particleDataWidth);
    }

    gl.bindTexture(gl.TEXTURE_2D, this.texEdgeStore);
    const storeRows = Math.floor(this.edgeCount / this.edgeStoreWidth);
    const storeRem = this.edgeCount % this.edgeStoreWidth;
    const edgeStoreFloat = new Float32Array(data.edgeStore);
    if (storeRows > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.edgeStoreWidth, storeRows, gl.RED, gl.FLOAT, edgeStoreFloat);
    }
    if (storeRem > 0) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, storeRows, storeRem, 1, gl.RED, gl.FLOAT, edgeStoreFloat, storeRows * this.edgeStoreWidth);
    }
    glValidate(gl, 'Data Uploads');

    // Initialize identity map (PID -> Physical Slot)
    this.kIdentity.run({ texIdMassTintEdgePtr: this.texIdMassTint });
    glValidate(gl, 'KIdentity Run');
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
    const doDebug = (now - this._lastDebugTime >= 20000) && /** @type {*} */(window).LOG_NEXT_FRAME;
    if (/** @type {*} */(window).LOG_NEXT_FRAME) {
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
