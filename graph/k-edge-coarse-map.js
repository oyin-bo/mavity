// @ts-check
import { createProgramSafe } from './utils.js';

/**
 * KEdgeCoarseMap
 * 
 * Generates an acceleration structure (Coarse Map) for edge lookups.
 * Maps `EdgeIndex / Stride` -> `ParticleIndex`.
 * 
 * Uses Binary Search over the monotonic `Start` indices in `texEdgePtr`.
 */
export class KEdgeCoarseMap {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   coarseWidth: number,
   *   coarseHeight: number,
   *   particleDataWidth: number,
   *   particleDataHeight: number,
   *   edgeCoarseMapStride: number,
   *   particleCount: number
   * }} options
   */
  constructor({
    gl,
    coarseWidth, coarseHeight,
    particleDataWidth, particleDataHeight,
    edgeCoarseMapStride,
    particleCount
  }) {
    this.gl = gl;
    this.coarseWidth = coarseWidth;
    this.coarseHeight = coarseHeight;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.stride = edgeCoarseMapStride;
    this.particleCount = particleCount;

    this.program = createProgramSafe({
      gl,
      vertexSource: VS_FULLSCREEN,
      fragmentSource: FS_COARSE_MAP
    });

    this.uEdgePtrLoc = gl.getUniformLocation(this.program, 'u_texEdgePtr');
    this.uDimsLoc = gl.getUniformLocation(this.program, 'u_particleDims');
    this.uStrideLoc = gl.getUniformLocation(this.program, 'u_stride');
    this.uCountLoc = gl.getUniformLocation(this.program, 'u_particleCount');
    this.uCoarseWidthLoc = gl.getUniformLocation(this.program, 'u_coarseWidth');

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.vbo = buf; // Keep reference to delete
    this.fbo = gl.createFramebuffer();
  }

  /**
   * @param {{
   *   texEdgePtr: WebGLTexture,
   *   texCoarseMap: WebGLTexture
   * }} input
   */
  run({ texEdgePtr, texCoarseMap }) {
    const gl = this.gl;

    gl.viewport(0, 0, this.coarseWidth, this.coarseHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    // Bind Input
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texEdgePtr);
    gl.uniform1i(this.uEdgePtrLoc, 0);

    // Uniforms
    gl.uniform2f(this.uDimsLoc, this.particleDataWidth, this.particleDataHeight);
    gl.uniform1i(this.uStrideLoc, this.stride);
    gl.uniform1i(this.uCountLoc, this.particleCount);
    gl.uniform1i(this.uCoarseWidthLoc, this.coarseWidth);

    // Bind Output
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texCoarseMap, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vbo);
    gl.deleteFramebuffer(this.fbo);
    gl.deleteProgram(this.program);
  }
}

const VS_FULLSCREEN = /*glsl*/`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FS_COARSE_MAP = /*glsl*/`#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D u_texEdgePtr; // .r = Start
uniform vec2 u_particleDims;
uniform int u_stride;
uniform int u_particleCount;
uniform int u_coarseWidth;

layout(location=0) out float o_particleIdx;

float getStart(int idx) {
    if (idx >= u_particleCount) return 1.0e15; // Infinity
    ivec2 coord = ivec2(
        idx % int(u_particleDims.x),
        idx / int(u_particleDims.x)
    );
    return texelFetch(u_texEdgePtr, coord, 0).r;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    int coarseIdx = coord.y * u_coarseWidth + coord.x;
    
    // We are looking for the particle that contains the edge at 'targetIndex'
    float targetIndex = float(coarseIdx) * float(u_stride);
    
    // Binary Search
    // Find P such that Start[P] <= targetIndex < Start[P+1]
    
    int low = 0;
    int high = u_particleCount; 
    
    // 17 iterations covers up to 131k particles.
    for (int i = 0; i < 17; i++) {
        int mid = (low + high) / 2;
        
        if (mid == low) {
            break; 
        }

        float val = getStart(mid);
        if (val <= targetIndex) {
            low = mid;
        } else {
            high = mid;
        }
    }
    
    o_particleIdx = float(low);
}`;

