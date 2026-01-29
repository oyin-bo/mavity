// @ts-check
import { createProgramSafe, glValidate } from './utils.js';
import { formatNumber } from '../gravity/diag.js';

/**
 * KParticleReshuffle Kernel
 * 
 * Reorders particle data according to the encoded sort atlas.
 * Uses Framebuffer with Multiple Render Targets (MRT) to write results.
 */

export class KParticleReshuffle {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   particleDataWidth: number,
   *   particleDataHeight: number,
   *   encodedSortOrderWidth: number,
   *   sortSpanSize: number
   * }} options
   */
  constructor({
    gl,
    particleCount,
    particleDataWidth, particleDataHeight,
    encodedSortOrderWidth,
    sortSpanSize
  }) {
    this.gl = gl;

    this.particleCount = particleCount;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.encodedSortOrderWidth = encodedSortOrderWidth;
    this.sortSpanSize = sortSpanSize;

    this.renderCount = 0;

    this.program = createProgramSafe({
      gl,
      vertexSource: /*glsl*/`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: /*glsl*/`#version 300 es
precision highp float;
precision highp usampler2D;

#define sortSpanSize ${this.sortSpanSize}
#define encodedSortOrderWidth ${this.encodedSortOrderWidth}
#define particleDataWidth ${this.particleDataWidth}
#define particleCount ${this.particleCount}

uniform sampler2D u_inParticlePositionAndSFC;
uniform sampler2D u_inParticleVelocity;
uniform sampler2D u_inIdMassTintEdgePtr;

uniform usampler2D u_inEncodedSortOrder0,
  u_inEncodedSortOrder1,
  u_inEncodedSortOrder2,
  u_inEncodedSortOrder3,
  u_inEncodedSortOrder4,
  u_inEncodedSortOrder5,
  u_inEncodedSortOrder6,
  u_inEncodedSortOrder7;
uniform int u_sortOffset;

layout(location=0) out vec4 o_positionAndSFC;
layout(location=1) out vec4 o_velocity;
layout(location=2) out vec4 o_idMassTintEdgePtr;

vec4 fetchPositionAndSFC(int i) {
    ivec2 coord = ivec2(i % particleDataWidth, i / particleDataWidth);
    return texelFetch(u_inParticlePositionAndSFC, coord, 0);
}
vec4 fetchVel(int i) {
    ivec2 coord = ivec2(i % particleDataWidth, i / particleDataWidth);
    return texelFetch(u_inParticleVelocity, coord, 0);
}
vec4 fetchIdMassTintEdgePtr(int i) {
    ivec2 coord = ivec2(i % particleDataWidth, i / particleDataWidth);
    return texelFetch(u_inIdMassTintEdgePtr, coord, 0);
}

uint fetchPackedFromOrder(int texIdx, ivec2 coord, int comp) {
    uvec4 ord;
    if (texIdx == 0) ord = texelFetch(u_inEncodedSortOrder0, coord, 0);
    else if (texIdx == 1) ord = texelFetch(u_inEncodedSortOrder1, coord, 0);
    else if (texIdx == 2) ord = texelFetch(u_inEncodedSortOrder2, coord, 0);
    else if (texIdx == 3) ord = texelFetch(u_inEncodedSortOrder3, coord, 0);
    else if (texIdx == 4) ord = texelFetch(u_inEncodedSortOrder4, coord, 0);
    else if (texIdx == 5) ord = texelFetch(u_inEncodedSortOrder5, coord, 0);
    else if (texIdx == 6) ord = texelFetch(u_inEncodedSortOrder6, coord, 0);
    else ord = texelFetch(u_inEncodedSortOrder7, coord, 0);
    if (comp == 0) return ord.x; else if (comp == 1) return ord.y; else if (comp == 2) return ord.z; return ord.w;
}

void main() {
    int idx = int(gl_FragCoord.y) * particleDataWidth + int(gl_FragCoord.x);
    if (idx >= particleCount) discard;

    int offset = u_sortOffset;
    int relIdx = idx - offset;

    if (relIdx < 0 || relIdx >= (particleCount - offset) / sortSpanSize * sortSpanSize) {
        o_positionAndSFC = fetchPositionAndSFC(idx);
        o_velocity = fetchVel(idx);
        o_idMassTintEdgePtr = fetchIdMassTintEdgePtr(idx);
        return;
    }

    int chunkIndex = relIdx / sortSpanSize;
    int localIdx = relIdx % sortSpanSize;
    ivec2 coord = ivec2(chunkIndex % encodedSortOrderWidth, chunkIndex / encodedSortOrderWidth);

    int packedIndex = localIdx >> 2; 
    int texIdx = packedIndex >> 2; 
    int comp = packedIndex & 3;

    uint packedVal = fetchPackedFromOrder(texIdx, coord, comp);
    uint byteShift = uint(localIdx & 3) * 8u;
    int srcLocal = int((packedVal >> byteShift) & 0xFFu);
    int srcGlobal = (chunkIndex * sortSpanSize + offset) + srcLocal;

    if (srcGlobal < 0 || srcGlobal >= particleCount) srcGlobal = idx;

    o_positionAndSFC = fetchPositionAndSFC(srcGlobal);
    o_velocity = fetchVel(srcGlobal);
    o_idMassTintEdgePtr = fetchIdMassTintEdgePtr(srcGlobal);
}
`
    });

    // Uniform locations cached
    this.uniforms = {
      u_inParticlePositionAndSFC: gl.getUniformLocation(this.program, 'u_inParticlePositionAndSFC'),
      u_inParticleVelocity: gl.getUniformLocation(this.program, 'u_inParticleVelocity'),
      u_inIdMassTintEdgePtr: gl.getUniformLocation(this.program, 'u_inIdMassTintEdgePtr'),
      u_sortOffset: gl.getUniformLocation(this.program, 'u_sortOffset'),
      u_inEncodedSortOrder0: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder0'),
      u_inEncodedSortOrder1: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder1'),
      u_inEncodedSortOrder2: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder2'),
      u_inEncodedSortOrder3: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder3'),
      u_inEncodedSortOrder4: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder4'),
      u_inEncodedSortOrder5: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder5'),
      u_inEncodedSortOrder6: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder6'),
      u_inEncodedSortOrder7: gl.getUniformLocation(this.program, 'u_inEncodedSortOrder7')
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
    /** @type {{outParticlePositionAndSFC: WebGLTexture|null, outParticleVelocity: WebGLTexture|null, outIdMassTintEdgePtr: WebGLTexture|null}} */
    this._fboShadow = { outParticlePositionAndSFC: null, outParticleVelocity: null, outIdMassTintEdgePtr: null };

    this.renderCount = 0;
  }

  /**
   * Run the reshuffle (synchronous)
   * @param {{
   *   outParticlePositionAndSFC: WebGLTexture,
   *   outParticleVelocity: WebGLTexture,
   *   outIdMassTintEdgePtr: WebGLTexture,
   *   inParticlePositionAndSFC: WebGLTexture,
   *   inParticleVelocity: WebGLTexture,
   *   inIdMassTintEdgePtr: WebGLTexture,
   *   inEncodedSortOrder: WebGLTexture[],
   *   sortOffset: number
   * }} params
   */
  run({
    outParticlePositionAndSFC,
    outParticleVelocity,
    outIdMassTintEdgePtr,
    inParticlePositionAndSFC,
    inParticleVelocity,
    inIdMassTintEdgePtr,
    inEncodedSortOrder,
    sortOffset
  }) {
    const gl = this.gl;
    glValidate(gl, 'KParticleReshuffle Run Start');

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
      glValidate(gl, 'KParticleReshuffle FBO Setup');
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

    for (let i = 0; i < 8; i++) {
      gl.activeTexture(gl.TEXTURE3 + i);
      gl.bindTexture(gl.TEXTURE_2D, inEncodedSortOrder[i]);
      // @ts-ignore
      gl.uniform1i(this.uniforms[`u_inEncodedSortOrder${i}`], 3 + i);
    }

    gl.uniform1i(this.uniforms.u_sortOffset, sortOffset);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    glValidate(gl, 'KParticleReshuffle Post-Draw');

    gl.bindVertexArray(null);
    for (let i = 0; i < 11; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
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
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      sortSpanSize: this.sortSpanSize,
      renderCount: this.renderCount,
      toString: () => this.toString()
    };
  }

  toString() {
    return `KParticleReshuffle(${this.particleCount}) span=${this.sortSpanSize} #${this.renderCount}`;
  }

  dispose() {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.outFramebuffer) gl.deleteFramebuffer(this.outFramebuffer);
  }
}
