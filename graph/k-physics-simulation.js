// @ts-check

/**
 * KPhysicsSimulation Kernel
 * 
 * Calculates forces and updates velocities based on the near-field SFC window.
 * Uses Framebuffer with Multiple Render Targets (MRT) to write results.
 * Follows the WebGL2 Robust Kernel contract.
 */

import { glValidate, createProgramSafe } from './utils.js';
import { readLinear, formatNumber } from '../gravity/diag.js';

export class KPhysicsSimulation {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   particleDataWidth: number,
   *   particleDataHeight: number,
   *   edgeStoreWidth: number,
   *   edgeStoreHeight: number,
   *   gravityWindow: number,
   *   sfcResolution: number
   *   G?: number,
   *   springK?: number,
   *   eps?: number
   * }} options
   */
  constructor({
    gl,
    particleCount,
    particleDataWidth, particleDataHeight,
    edgeStoreWidth, edgeStoreHeight,
    gravityWindow,
    sfcResolution,
    G,
    springK,
    eps
  }) {
    this.gl = gl;

    this.particleCount = particleCount;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.edgeStoreWidth = edgeStoreWidth;
    this.edgeStoreHeight = edgeStoreHeight;
    this.gravityWindow = gravityWindow;
    this.sfcResolution = sfcResolution;
    // Default physics params
    this.G = G || 1.0;
    this.springK = springK || 1.0;
    this.eps = eps || 0.0001;

    this.renderCount = 0;

    this.program = createProgramSafe({
      gl,
      vertexSource: /* glsl */`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: /* glsl */`#version 300 es
precision highp float;

#define WINDOW ${this.gravityWindow}
#define SFC_S ${this.sfcResolution.toFixed(1)}
#define EDGE_STORE_W ${this.edgeStoreWidth}
#define PARTICLE_DATA_W ${this.particleDataWidth}
#define PARTICLE_COUNT ${this.particleCount}

uniform sampler2D u_inParticlePositionAndSFC;
uniform sampler2D u_inParticleVelocity;
uniform sampler2D u_inIdMassTintEdgePtr;
uniform sampler2D u_texEdgePtr;
uniform sampler2D u_texEdgeStore;

uniform float u_dt;
uniform float u_G;
uniform float u_springK;
uniform float u_eps;

layout(location=0) out vec4 o_positionAndSFC;
layout(location=1) out vec4 o_velocity;
layout(location=2) out vec4 o_idMassTintEdgePtr;

uint hash(uint x) {
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

float getSFCKey(vec3 p) {
  // 1. Octahedral Projection (Butterfly Map)
  float l1 = abs(p.x) + abs(p.y) + abs(p.z) + 1e-8;
  vec2 u = p.xy / l1;
  if (p.z < 0.0) {
      vec2 s = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
      u = (1.0 - abs(u.yx)) * s;
  }
  uvec2 coord = uvec2((u * 0.5 + 0.5) * 2047.0);

  // 2. Hierarchical Stochastic Hilbert
  // We hash based on the PARENT cell at each bit level.
  // This breaks Moiré across all scales while preserving spatial locality.
  uint state = 0u;
  uint d = 0u;
  for (int i = 10; i >= 0; i--) {
      uint qx = (coord.x >> i) & 1u;
      uint qy = (coord.y >> i) & 1u;
      
      // Hash of parent coordinates (bits above current i)
      uint h = hash((coord.x >> (i + 1)) ^ (coord.y >> (i + 2)) ^ uint(i)) & 3u;
      uint effective_state = state ^ h;

      uint quad;
      uint next_state;
      if (effective_state == 0u) { // A
          if (qx == 0u && qy == 0u) { quad = 0u; next_state = 1u; }
          else if (qx == 0u && qy == 1u) { quad = 1u; next_state = 0u; }
          else if (qx == 1u && qy == 1u) { quad = 2u; next_state = 0u; }
          else { quad = 3u; next_state = 3u; }
      } else if (effective_state == 1u) { // B
          if (qx == 0u && qy == 0u) { quad = 0u; next_state = 0u; }
          else if (qx == 1u && qy == 0u) { quad = 1u; next_state = 1u; }
          else if (qx == 1u && qy == 1u) { quad = 2u; next_state = 1u; }
          else { quad = 3u; next_state = 2u; }
      } else if (effective_state == 2u) { // C
          if (qx == 1u && qy == 1u) { quad = 0u; next_state = 1u; }
          else if (qx == 1u && qy == 0u) { quad = 1u; next_state = 2u; }
          else if (qx == 0u && qy == 0u) { quad = 2u; next_state = 2u; }
          else { quad = 3u; next_state = 3u; }
      } else { // D
          if (qx == 1u && qy == 1u) { quad = 0u; next_state = 2u; }
          else if (qx == 0u && qy == 1u) { quad = 1u; next_state = 3u; }
          else if (qx == 0u && qy == 0u) { quad = 2u; next_state = 3u; }
          else { quad = 3u; next_state = 0u; }
      }
      
      d = (d << 2u) | quad;
      state = next_state;
  }
  
  // 3. Final Key
  float tie = (p.x + p.y + p.z + 5.0) * 1.0e-9; 
  return (float(d) / 4194304.0) + tie;
}

vec4 fetchPos(int i) {
  uint ii = uint((i + PARTICLE_COUNT) % PARTICLE_COUNT);
  ivec2 coord = ivec2(int(ii % uint(PARTICLE_DATA_W)), int(ii / uint(PARTICLE_DATA_W)));
  return texelFetch(u_inParticlePositionAndSFC, coord, 0);
}

float fetchNextEdgePtr(int idx) {
  int nextIdx = idx + 1;
  ivec2 coord = ivec2(nextIdx % PARTICLE_DATA_W, nextIdx / PARTICLE_DATA_W);
  return texelFetch(u_texEdgePtr, coord, 0).r;
}

void main() {
  ivec2 texelCoord = ivec2(gl_FragCoord.xy);
  int idx = texelCoord.y * PARTICLE_DATA_W + texelCoord.x;
  
  if (idx >= PARTICLE_COUNT) {
    discard;
    return;
  }

  vec4 positionAndSFC = texelFetch(u_inParticlePositionAndSFC, texelCoord, 0);
  vec4 velocity = texelFetch(u_inParticleVelocity, texelCoord, 0);
  vec4 idMassTintEdgePtr = texelFetch(u_inIdMassTintEdgePtr, texelCoord, 0);

  vec3 p = positionAndSFC.xyz;
  vec3 v = velocity.xyz;
  vec3 acc = vec3(0.0);
  
  // Near-field REPULSIVE gravity
  for (int i = -WINDOW; i <= WINDOW; i++) {
    if (i == 0) continue;
    int targetIdx = idx + i;
    if (targetIdx < 0 || targetIdx >= PARTICLE_COUNT) continue;
    
    vec4 otherA = fetchPos(targetIdx); 
    
    vec3 dp = otherA.xyz - p;
    float d2 = dot(dp, dp) + u_eps; 
    float inv = inversesqrt(d2);
    float inv3 = inv * inv * inv;
    
    // REPULSIVE: subtract force instead of adding
    acc -= u_G * dp * inv3;
  }

  // Laplacian Edge Pull
  int edgeStart = int(texelFetch(u_texEdgePtr, texelCoord, 0).r);
  int edgeEnd = int(fetchNextEdgePtr(idx));
  for (int e = edgeStart; e < edgeEnd; e++) {
    ivec2 storeCoord = ivec2(e % EDGE_STORE_W, e / EDGE_STORE_W);
    int targetPhys = int(texelFetch(u_texEdgeStore, storeCoord, 0).r);
    if (targetPhys >= 0) {
      vec4 targetA = fetchPos(targetPhys);
      acc += (targetA.xyz - p) * u_springK; // Laplacian pull
    }
  }
    
  // Core physics stabilization
  float r = length(p);
  if (r > 0.0001) {
    acc -= (p / r) * (0.000005 * clamp(r - 1.2, -0.5, 2.0)); 
  }


  // Semi-implicit Euler integration
  float damp = 0.998;
  vec3 v_next = (v + acc * u_dt) * damp;
  vec3 p_next = p + v_next * u_dt;
  
  o_positionAndSFC = vec4(p_next, getSFCKey(p_next)); 
  o_velocity = vec4(v_next, velocity.w);
  o_idMassTintEdgePtr = idMassTintEdgePtr;
}
`
    });

    // Uniform locations
    this.uniforms = {
      u_inParticlePositionAndSFC: gl.getUniformLocation(this.program, 'u_inParticlePositionAndSFC'),
      u_inParticleVelocity: gl.getUniformLocation(this.program, 'u_inParticleVelocity'),
      u_inIdMassTintEdgePtr: gl.getUniformLocation(this.program, 'u_inIdMassTintEdgePtr'),
      u_texEdgePtr: gl.getUniformLocation(this.program, 'u_texEdgePtr'),
      u_texEdgeStore: gl.getUniformLocation(this.program, 'u_texEdgeStore'),
      u_dt: gl.getUniformLocation(this.program, 'u_dt'),
      u_G: gl.getUniformLocation(this.program, 'u_G'),
      u_springK: gl.getUniformLocation(this.program, 'u_springK'),
      u_eps: gl.getUniformLocation(this.program, 'u_eps')
    };

    // Self-contained VAO
    this.quadVAO = gl.createVertexArray();
    this.quadBuf = gl.createBuffer();
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Internal FBO for MRT
    this.outFramebuffer = gl.createFramebuffer();

    /** @type {{ outParticlePositionAndSFC: WebGLTexture | null, outParticleVelocity: WebGLTexture | null, outIdMassTintEdgePtr: WebGLTexture | null }} */
    this._fboShadow = { outParticlePositionAndSFC: null, outParticleVelocity: null, outIdMassTintEdgePtr: null };

    this.renderCount = 0;
  }

  /**
   * Run the kernel (synchronous)
   * @param {{
   *   outParticlePositionAndSFC: WebGLTexture,
   *   outParticleVelocity: WebGLTexture,
   *   outIdMassTintEdgePtr: WebGLTexture,
   *   inParticlePositionAndSFC: WebGLTexture,
   *   inParticleVelocity: WebGLTexture,
   *   inIdMassTintEdgePtr: WebGLTexture,
   *   inTexEdgePtr: WebGLTexture,
   *   inTexEdgeStore: WebGLTexture,
   *   dt: number
   * }} params
   */
  run({
    outParticlePositionAndSFC,
    outParticleVelocity,
    outIdMassTintEdgePtr,
    inParticlePositionAndSFC,
    inParticleVelocity,
    inIdMassTintEdgePtr,
    inTexEdgePtr,
    inTexEdgeStore,
    dt
  }) {
    const gl = this.gl;
    glValidate(gl, 'KPhysics Run Start');

    gl.useProgram(this.program);

    // MRT Lazy Synchronization
    if (!this._fboShadow ||
      this._fboShadow.outParticlePositionAndSFC !== outParticlePositionAndSFC ||
      this._fboShadow.outParticleVelocity !== outParticleVelocity ||
      this._fboShadow.outIdMassTintEdgePtr !== outIdMassTintEdgePtr) {

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFramebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outParticlePositionAndSFC, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, outParticleVelocity, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, outIdMassTintEdgePtr, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      this._fboShadow = { outParticlePositionAndSFC, outParticleVelocity, outIdMassTintEdgePtr };
      glValidate(gl, 'KPhysics FBO Setup');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFramebuffer);
    gl.viewport(0, 0, this.particleDataWidth, this.particleDataHeight);

    // Isolation
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.depthMask(false);

    // Inputs
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inParticlePositionAndSFC);
    gl.uniform1i(this.uniforms.u_inParticlePositionAndSFC, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inParticleVelocity);
    gl.uniform1i(this.uniforms.u_inParticleVelocity, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inIdMassTintEdgePtr);
    gl.uniform1i(this.uniforms.u_inIdMassTintEdgePtr, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, inTexEdgePtr);
    gl.uniform1i(this.uniforms.u_texEdgePtr, 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, inTexEdgeStore);
    gl.uniform1i(this.uniforms.u_texEdgeStore, 4);

    gl.uniform1f(this.uniforms.u_dt, dt);
    gl.uniform1f(this.uniforms.u_G, this.G);
    gl.uniform1f(this.uniforms.u_springK, this.springK);
    gl.uniform1f(this.uniforms.u_eps, this.eps);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    glValidate(gl, 'KPhysics Post-Draw');

    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);

    this.renderCount++;
  }

  /**
   * @param {{ pixels?: boolean }} [options]
   */
  valueOf({ pixels } = {}) {
    return {
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
      renderCount: this.renderCount,
      toString: () => this.toString()
    };
  }

  toString() {
    return `KPhysicsSimulation(${this.particleCount}) G=${formatNumber(this.G)} window=${this.gravityWindow} #${this.renderCount}`;
  }

  dispose() {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.outFramebuffer) gl.deleteFramebuffer(this.outFramebuffer);
  }
}
