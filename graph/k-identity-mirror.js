// @ts-check

import { createProgramSafe } from './utils.js';

/**
 * KIdentityMirror Kernel
 * 
 * Maintains the identity map (phonebook) between persistent IDs and their current physical indices.
 * Allows looking up particles by their ID even after sorting.
 */
export class KIdentityMirror {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   textureWidth: number,
   *   textureHeight: number,
   * }} options
   */
  constructor({
    gl,
    particleCount,
    textureWidth, textureHeight,
  }) {
    this.gl = gl;
    this.particleCount = particleCount;
    this.textureWidth = textureWidth;
    this.textureHeight = textureHeight;

    this.vaoEmpty = gl.createVertexArray(); // Used for vertex-pulling kernels to avoid poisoning state
    gl.bindVertexArray(null);

    this.texIdentity = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texIdentity);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.textureWidth, this.textureHeight, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    this.identityFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.identityFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texIdentity, 0);

    this.program = createProgramSafe({
      gl,
      vertexSource: /* glsl */`#version 300 es
uniform sampler2D u_texIdMassTintEdgePtr;
uniform float u_particleDataWidth;
uniform vec2 u_res;
flat out float v_physID;
void main() {
    int pIndex = gl_VertexID;
    int tx = pIndex % int(u_particleDataWidth);
    int ty = pIndex / int(u_particleDataWidth);

    vec4 idMassTintEdgePtr = texelFetch(u_texIdMassTintEdgePtr, ivec2(tx, ty), 0);
    float pid = floor(idMassTintEdgePtr.x + 0.5);

    v_physID = float(pIndex);
    float x = mod(pid, u_res.x);
    float y = floor(pid / u_res.x);
    gl_Position = vec4((x + 0.5) / u_res.x * 2.0 - 1.0, (y + 0.5) / u_res.y * 2.0 - 1.0, 0.0, 1.0);
    gl_Position.z = 0.0;
    gl_Position.w = 1.0;
    gl_PointSize = 1.0;
}
`,
      fragmentSource: /* glsl */`#version 300 es
precision highp float;
flat in float v_physID;
layout(location=0) out float o_val;
void main() { 
    // Optimization: avoid interpolation noise if any, although flat should handle it.
    o_val = v_physID; 
}
`
    });

  }

  /**
   * @param {{
   *   texIdMassTintEdgePtr: WebGLTexture
   * }} params
   */
  run({ texIdMassTintEdgePtr }) {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.identityFbo);
    this.gl.viewport(0, 0, this.textureWidth, this.textureHeight);
    this.gl.clearBufferfv(this.gl.COLOR, 0, new Float32Array([-1.0, 0, 0, 0]));

    this.gl.bindVertexArray(this.vaoEmpty);
    this.gl.useProgram(this.program);

    this.gl.activeTexture(this.gl.TEXTURE0);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texIdMassTintEdgePtr);
    this.gl.uniform1i(this.gl.getUniformLocation(this.program, 'u_texIdMassTintEdgePtr'), 0);
    this.gl.uniform1f(this.gl.getUniformLocation(this.program, 'u_particleDataWidth'), this.textureWidth);

    this.gl.uniform2f(this.gl.getUniformLocation(this.program, 'u_res'), this.textureWidth, this.textureHeight);

    this.gl.drawArrays(this.gl.POINTS, 0, this.particleCount);

    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.useProgram(null);
  }

  dispose() {
    if (this.texIdentity) this.gl.deleteTexture(this.texIdentity);
    if (this.identityFbo) this.gl.deleteFramebuffer(this.identityFbo);
    this.gl.deleteProgram(this.program);
  }
}
