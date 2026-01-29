// @ts-check

/**
 * KPhysicsSentinel - Robust Debug probe kernel
 * 
 * Writes known sentinel values to ALL MRT targets.
 * Follows the WebGL2 Robust Kernel contract.
 */

import { glValidate, createProgramSafe } from './utils.js';
import { formatNumber } from '../gravity/diag.js';

export class KPhysicsSentinel {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleDataWidth: number,
   *   particleDataHeight: number
   * }} options
   */
  constructor(options) {
    this.gl = options.gl;
    const gl = this.gl;

    this.width = options.particleDataWidth;
    this.height = options.particleDataHeight;

    this.program = createProgramSafe({
      gl,
      vertexSource: `#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: `#version 300 es
precision highp float;
layout(location=0) out vec4 o_pos;
layout(location=1) out vec4 o_vel;
layout(location=2) out vec4 o_meta;
void main() {
  o_pos = vec4(1001.0, 1002.0, 1003.0, 1004.0);
  o_vel = vec4(2001.0, 2002.0, 2003.0, 2004.0);
  o_meta = vec4(3001.0, 3002.0, 3003.0, 4001.0);
}`
    });

    // Self-contained VAO
    this.quadVAO = gl.createVertexArray();
    this.quadBuf = gl.createBuffer();
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Internal FBO
    this.outFramebuffer = gl.createFramebuffer();

    /** @type {{ outParticlePositionAndSFC: WebGLTexture | null, outParticleVelocity: WebGLTexture | null, outIdMassTintEdgePtr: WebGLTexture | null }} */
    this._fboShadow = { outParticlePositionAndSFC: null, outParticleVelocity: null, outIdMassTintEdgePtr: null };

    this.renderCount = 0;
  }

  /**
   * @param {{
   *   outParticlePositionAndSFC: WebGLTexture,
   *   outParticleVelocity: WebGLTexture,
   *   outIdMassTintEdgePtr: WebGLTexture
   * }} params
   */
  run({ outParticlePositionAndSFC, outParticleVelocity, outIdMassTintEdgePtr }) {
    const gl = this.gl;
    glValidate(gl, 'Sentinel Run Start');

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
      glValidate(gl, 'Sentinel FBO Setup');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFramebuffer);
    gl.viewport(0, 0, this.width, this.height);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.depthMask(false);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    glValidate(gl, 'Sentinel Post-Draw');

    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);

    this.renderCount++;
  }

  /**
   * @param {{ pixels?: boolean }} [options]
   */
  valueOf({ pixels } = {}) {
    return {
      width: this.width,
      height: this.height,
      renderCount: this.renderCount,
      toString: () => this.toString()
    };
  }

  toString() {
    return `KPhysicsSentinel(${this.width}x${this.height}) #${this.renderCount}`;
  }

  dispose() {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.outFramebuffer) gl.deleteFramebuffer(this.outFramebuffer);
  }
}
