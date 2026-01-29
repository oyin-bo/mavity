// @ts-check

/**
 * KSortEncoder Kernel
 * 
 * Performs a GPU bitonic sort in chunks to maintain spatial locality.
 * Generates an order registry used to reorder particles.
 * Follows the WebGL2 Robust Kernel contract.
 */

import { glValidate, createProgramSafe } from './utils.js';
import { readLinear, formatNumber } from '../gravity/diag.js';

export class KSortEncoder {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   particleCount: number,
   *   particleDataWidth: number,
   *   particleDataHeight: number,
   *   sortSpanSize: number,
   *   encodedSortOrderWidth: number,
   *   encodedSortOrderHeight: number
   * }} options
   */
  constructor({
    gl,
    particleCount,
    particleDataWidth, particleDataHeight,
    sortSpanSize,
    encodedSortOrderWidth, encodedSortOrderHeight
  }) {
    this.gl = gl;

    this.particleCount = particleCount;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.sortSpanSize = sortSpanSize;
    this.encodedSortOrderWidth = encodedSortOrderWidth;
    this.encodedSortOrderHeight = encodedSortOrderHeight;

    this.renderCount = 0;

    this.program = createProgramSafe({
      gl,
      vertexSource: /* glsl */`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`,
      fragmentSource: /* glsl */`#version 300 es
precision highp float;
precision highp sampler2D;

#define CHUNK ${this.sortSpanSize}
#define DATA_W ${this.particleDataWidth}
#define ENC_W ${this.encodedSortOrderWidth}
#define N_COUNT ${this.particleCount}

uniform sampler2D u_texParticlePositionAndSFC; 
uniform int u_sortOffset; 

layout(location=0) out uvec4 o0;
layout(location=1) out uvec4 o1;
layout(location=2) out uvec4 o2;
layout(location=3) out uvec4 o3;
layout(location=4) out uvec4 o4;
layout(location=5) out uvec4 o5;
layout(location=6) out uvec4 o6;
layout(location=7) out uvec4 o7;

float fetchSFC(int i) {
    int ii = (i + N_COUNT) % N_COUNT;
    return texelFetch(u_texParticlePositionAndSFC, ivec2(ii % DATA_W, ii / DATA_W), 0).w;
}

void main() {
    ivec2 pix = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y));
    int chunkIndex = pix.y * ENC_W + pix.x; 
    int base = (chunkIndex * CHUNK + u_sortOffset);

    float keys[CHUNK];
    int ids[CHUNK];
    for (int i = 0; i < CHUNK; i++) {
        int gIdx = base + i;
        if (gIdx >= 0 && gIdx < N_COUNT) {
            keys[i] = fetchSFC(gIdx);
        } else {
            keys[i] = 1.0e37; 
        }
        ids[i] = i;
    }

    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < CHUNK; i++) {
                int l = i ^ j;
                if (l > i) {
                    bool dir = (i & k) == 0;
                    if ((keys[i] > keys[l]) == dir) {
                        float tk = keys[i]; keys[i] = keys[l]; keys[l] = tk;
                        int ti = ids[i]; ids[i] = ids[l]; ids[l] = ti;
                    }
                }
            }
        }
    }

    uint packed[32];
    for (int i = 0; i < 32; i++) {
        uint a = uint(ids[i*4 + 0]) & 0xFFu;
        uint b = uint(ids[i*4 + 1]) & 0xFFu;
        uint c = uint(ids[i*4 + 2]) & 0xFFu;
        uint d = uint(ids[i*4 + 3]) & 0xFFu;
        packed[i] = a | (b << 8u) | (c << 16u) | (d << 24u);
    }

    o0 = uvec4(packed[0], packed[1], packed[2], packed[3]);
    o1 = uvec4(packed[4], packed[5], packed[6], packed[7]);
    o2 = uvec4(packed[8], packed[9], packed[10], packed[11]);
    o3 = uvec4(packed[12], packed[13], packed[14], packed[15]);
    o4 = uvec4(packed[16], packed[17], packed[18], packed[19]);
    o5 = uvec4(packed[20], packed[21], packed[22], packed[23]);
    o6 = uvec4(packed[24], packed[25], packed[26], packed[27]);
    o7 = uvec4(packed[28], packed[29], packed[30], packed[31]);
}
`
    });

    // Uniform locations
    this.uniforms = {
      u_texParticlePositionAndSFC: gl.getUniformLocation(this.program, 'u_texParticlePositionAndSFC'),
      u_sortOffset: gl.getUniformLocation(this.program, 'u_sortOffset')
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

    // Internal FBO
    this.outFramebuffer = gl.createFramebuffer();
    this._fboShadow = null;
    this.drawBufs = Array.from({ length: 8 }, (_, i) => gl.COLOR_ATTACHMENT0 + i);

    this.renderCount = 0;
  }

  /**
   *   texParticlePositionAndSFC: WebGLTexture,
   *   outTexEncodedOrder: WebGLTexture[],
   *   sortOffset: number
   * }} params
   */
  run({ texParticlePositionAndSFC, outTexEncodedOrder, sortOffset }) {
    const gl = this.gl;
    glValidate(gl, 'KSortEncoder Run Start');

    gl.useProgram(this.program);

    // MRT Lazy Synchronization
    if (this._fboShadow !== outTexEncodedOrder[0]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFramebuffer);
      for (let i = 0; i < 8; i++) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, outTexEncodedOrder[i], 0);
      }
      gl.drawBuffers(this.drawBufs);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this._fboShadow = outTexEncodedOrder[0];
      glValidate(gl, 'KSortEncoder FBO Setup');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFramebuffer);
    gl.viewport(0, 0, this.encodedSortOrderWidth, this.encodedSortOrderHeight);

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
    gl.bindTexture(gl.TEXTURE_2D, texParticlePositionAndSFC);
    gl.uniform1i(this.uniforms.u_texParticlePositionAndSFC, 0);

    gl.uniform1i(this.uniforms.u_sortOffset, sortOffset);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    glValidate(gl, 'KSortEncoder Post-Draw');

    gl.bindVertexArray(null);
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
      sortSpanSize: this.sortSpanSize,
      encodedSortOrderWidth: this.encodedSortOrderWidth,
      encodedSortOrderHeight: this.encodedSortOrderHeight,
      renderCount: this.renderCount,
      toString: () => this.toString()
    };
  }

  toString() {
    return `KSortEncoder(${this.particleCount}) span=${this.sortSpanSize} #${this.renderCount}`;
  }

  dispose() {
    const gl = this.gl;
    if (this.program) gl.deleteProgram(this.program);
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.outFramebuffer) gl.deleteFramebuffer(this.outFramebuffer);
  }
}
