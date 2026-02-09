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
   *   eps?: number,
   *   damping?: number
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
    eps,
    damping
  }) {
    this.gl = gl;

    this.particleCount = particleCount;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.edgeStoreWidth = edgeStoreWidth;
    this.edgeStoreHeight = edgeStoreHeight;
    this.gravityWindow = gravityWindow;
    this.sfcResolution = sfcResolution;
    // Default physics params (damping is compile-time constant defaulted to 0)
    this.G = G !== undefined ? G : 1.0;
    this.springK = springK !== undefined ? springK : 1.0;
    this.eps = eps !== undefined ? eps : 0.0001;
    this.damping = damping !== undefined ? damping : 0.0;

    this.renderCount = 0;

    this.program = createProgramSafe({
      gl,
      vertexSource: /* glsl */`#version 300 es
layout(location = 0) in int a_idx;
flat out int vIdx;
#define DATA_W ${this.particleDataWidth}
#define DATA_H ${this.particleDataHeight}
void main() {
  vIdx = a_idx;
  float x = (float(vIdx % DATA_W) + 0.5) / float(DATA_W) * 2.0 - 1.0;
  float y = (float(vIdx / DATA_W) + 0.5) / float(DATA_H) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;
}`,
      fragmentSource: /* glsl */`#version 300 es
precision highp float;

#define WINDOW ${this.gravityWindow}
#define SFC_S ${this.sfcResolution.toExponential()}
#define DATA_W ${this.particleDataWidth}
#define PARTICLE_COUNT ${this.particleCount}
#define EDGE_STORE_W ${this.edgeStoreWidth}

flat in int vIdx;

uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uMass;
uniform sampler2D uEdgePtr;
uniform sampler2D uEdgeStore;

uniform float u_dt;
uniform float u_G;
uniform float u_springK;
uniform float u_eps;
uniform float u_damping;

layout(location=0) out vec4 oPos;
layout(location=1) out vec4 oVel;
layout(location=2) out vec4 oMass;

float getSFCKey(vec3 p) {
  float l1 = abs(p.x) + abs(p.y) + abs(p.z) + 1e-20;
  vec2 uv = p.xy / l1;
  if (p.z < 0.0) {
    uv = (1.0 - abs(uv.yx)) * vec2(uv.x >= 0.0 ? 1.0 : -1.0, uv.y >= 0.0 ? 1.0 : -1.0);
  }
  return uv.x + uv.y * 2.0;
}

vec4 fetchPos(int i) {
  ivec2 coord = ivec2(i % DATA_W, i / DATA_W);
  return texelFetch(uPos, coord, 0);
}

float fetchMass(int i) {
  ivec2 coord = ivec2(i % DATA_W, i / DATA_W);
  return texelFetch(uMass, coord, 0).y;
}

float fetchNextEdgePtr(int idx) {
  int nextIdx = idx + 1;
  ivec2 coord = ivec2(nextIdx % DATA_W, nextIdx / DATA_W);
  return texelFetch(uEdgePtr, coord, 0).r;
}

void main() {
  int idx = vIdx;
  if (idx >= int(PARTICLE_COUNT)) { discard; return; }
  ivec2 coord = ivec2(idx % DATA_W, idx / DATA_W);

  vec4 posSFC = texelFetch(uPos, coord, 0);
  vec4 vel = texelFetch(uVel, coord, 0);
  vec4 idMass = texelFetch(uMass, coord, 0);

  vec3 p = posSFC.xyz;
  vec3 v = vel.xyz;
  vec3 acc = vec3(0.0);

  // Near-field Gravity
  for (int i = -WINDOW; i <= WINDOW; i++) {
    if (i == 0) continue;
    int t = idx + i;
    if (t < 0 || t >= int(PARTICLE_COUNT)) continue;
    
    vec4 otherA = fetchPos(t);
    vec3 dp = otherA.xyz - p;
    float d2 = dot(dp, dp) + u_eps;
    float inv = 1.0 / sqrt(d2);
    float inv3 = inv * inv * inv;
    float GM = u_G * fetchMass(t);
    acc += dp * (GM * inv3);
  }

  // Laplacian Edge Pull
  int edgeStart = int(texelFetch(uEdgePtr, coord, 0).r);
  int edgeEnd = int(fetchNextEdgePtr(idx));
  for (int e = edgeStart; e < edgeEnd; e++) {
    ivec2 storeCoord = ivec2(e % EDGE_STORE_W, e / EDGE_STORE_W);
    int targetPhys = int(texelFetch(uEdgeStore, storeCoord, 0).r);
    if (targetPhys >= 0) {
      vec4 targetA = fetchPos(targetPhys);
      acc += (targetA.xyz - p) * u_springK; // Laplacian pull
    }
  }

  // Integration
  vec3 v_next = (v + acc * u_dt) * (1.0 - u_damping);
  vec3 p_next = p + v_next * u_dt;

  oPos = vec4(p_next, getSFCKey(p_next)); 
  oVel = vec4(v_next, vel.w);
  oMass = idMass;
}
`
    });

    // Uniform locations
    this.uniforms = {
      uPos: gl.getUniformLocation(this.program, 'uPos'),
      uVel: gl.getUniformLocation(this.program, 'uVel'),
      uMass: gl.getUniformLocation(this.program, 'uMass'),
      uEdgePtr: gl.getUniformLocation(this.program, 'uEdgePtr'),
      uEdgeStore: gl.getUniformLocation(this.program, 'uEdgeStore'),
      u_dt: gl.getUniformLocation(this.program, 'u_dt'),
      u_G: gl.getUniformLocation(this.program, 'u_G'),
      u_springK: gl.getUniformLocation(this.program, 'u_springK'),
      u_eps: gl.getUniformLocation(this.program, 'u_eps'),
      u_damping: gl.getUniformLocation(this.program, 'u_damping')
    };

    // Self-contained VAO for particles (using index attribute for identity)
    this.quadVAO = gl.createVertexArray();
    this.quadBuf = gl.createBuffer();
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    const indices = new Int32Array(this.particleCount);
    for (let i = 0; i < this.particleCount; i++) indices[i] = i;
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 1, gl.INT, 0, 0); 
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
    gl.uniform1i(this.uniforms.uPos, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inParticleVelocity);
    gl.uniform1i(this.uniforms.uVel, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inIdMassTintEdgePtr);
    gl.uniform1i(this.uniforms.uMass, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, inTexEdgePtr);
    gl.uniform1i(this.uniforms.uEdgePtr, 3);

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, inTexEdgeStore);
    gl.uniform1i(this.uniforms.uEdgeStore, 4);

    gl.uniform1f(this.uniforms.u_dt, dt);
    gl.uniform1f(this.uniforms.u_G, this.G);
    gl.uniform1f(this.uniforms.u_springK, this.springK);
    gl.uniform1f(this.uniforms.u_eps, this.eps);
    gl.uniform1f(this.uniforms.u_damping, this.damping);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
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

  valueOf() {
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
