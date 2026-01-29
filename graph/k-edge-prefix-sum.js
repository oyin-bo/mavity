// @ts-check
import { createProgramSafe } from './utils.js';
import { formatNumber } from '../gravity/diag.js';

/**
 * KEdgePrefixSum
 * 
 * Performs a Hillis-Steele prefix sum on the edge counts stored in the Metadata texture.
 * Encapsulates its own temporary buffers for the scan process.
 * 
 * Output: texEdgePtr (R32F, .r = start index)
 */
export class KEdgePrefixSum {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   width: number,
   *   height: number,
   *   encodedSortOrderWidth: number,
   *   particleCount: number
   * }} options
   */
  constructor({ gl, width, height, encodedSortOrderWidth, particleCount }) {
    this.gl = gl;
    this.width = width;
    this.height = height;
    this.particleCount = particleCount;

    this.renderCount = 0;

    // Internal Ping-Pong Textures for the Scan (R32UI)
    // We scan the 'Count' to produce 'Inclusive Sum'
    this.texSum = [this._createTexture(), this._createTexture()];
    this.fbo = gl.createFramebuffer();

    // -- Programs --

    // 1. Init: Extract Count from Meta -> Sum[0]
    this.progInit = createProgramSafe({
      gl,
      vertexSource: /*glsl*/`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: /*glsl*/`#version 300 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

#define encodedSortOrderWidth ${encodedSortOrderWidth}
#define particleCount ${particleCount}

uniform sampler2D u_inIdMassTintEdgePtr; // RGBA32F (PID, Mass, Tint, EdgePtrStart)
uniform sampler2D u_texEdgePtrOld;      // R32F
uniform usampler2D u_order0, u_order1, u_order2, u_order3, u_order4, u_order5, u_order6, u_order7;
uniform int u_sortOffset;
uniform vec2 u_dimensions;

layout(location=0) out float o_count;

uint fetchPackedFromOrder(int texIdx, ivec2 coord, int comp) {
    uvec4 ord;
    if (texIdx == 0) ord = texelFetch(u_order0, coord, 0);
    else if (texIdx == 1) ord = texelFetch(u_order1, coord, 0);
    else if (texIdx == 2) ord = texelFetch(u_order2, coord, 0);
    else if (texIdx == 3) ord = texelFetch(u_order3, coord, 0);
    else if (texIdx == 4) ord = texelFetch(u_order4, coord, 0);
    else if (texIdx == 5) ord = texelFetch(u_order5, coord, 0);
    else if (texIdx == 6) ord = texelFetch(u_order6, coord, 0);
    else ord = texelFetch(u_order7, coord, 0);
    if (comp == 0) return ord.x; else if (comp == 1) return ord.y; else if (comp == 2) return ord.z; return ord.w;
}

uint getOldID(uint pNew) {
    int idx = int(pNew);
    int offset = u_sortOffset;
    int relIdx = idx - offset;

    if (relIdx < 0 || relIdx >= (particleCount - offset) / 128 * 128) {
        return uint(idx);
    }

    int chunkIdx = relIdx / 128; // 128 items per chunk
    int sub = relIdx % 128;
    
    int texIdx = sub / 16; // 16 items per texture (128/8)
    int rem = sub % 16;
    int comp = rem / 4;    // 4 items per component
    int byteIdx = rem % 4; // 1 byte per item

    ivec2 coord = ivec2(chunkIdx % encodedSortOrderWidth, chunkIdx / encodedSortOrderWidth);
    uint val = fetchPackedFromOrder(texIdx, coord, comp);
    uint rel = (val >> (uint(byteIdx) * 8u)) & 0xFFu;
    
    return uint(chunkIdx * 128 + offset) + rel;
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int flatIdx = coord.y * int(u_dimensions.x) + coord.x;

  if (flatIdx >= int(particleCount)) {
      o_count = 0.0;
      return;
  }

  // Current Particle (in New Layout) -> Old Physical ID
  uint pOld = getOldID(uint(flatIdx));

  // Determine Count by looking at Old Edge Pointers
  // Count = Start[pOld + 1] - Start[pOld]
  // Note: This relies on Old Pointers being contiguous and monotonic in the OLD layout
  // But wait, the texture contains Pointers (Start).
  // The original seeding guarantees monotonicity.
  // We need to fetch Start[pOld] and Start[pOld+1].
  
  ivec2 coordOld = ivec2(int(pOld % uint(u_dimensions.x)), int(pOld / uint(u_dimensions.x)));
  float startCurrent = floor(texelFetch(u_texEdgePtrOld, coordOld, 0).r + 0.5);
  
  // For the last particle, we need a sentinel or total count.
  // Assuming the texture is padded or sentinel is injected at N.
  // If pOld is N-1, we read N.
  
  uint pNextIdx = pOld + 1u;
  ivec2 coordNext = ivec2(int(pNextIdx % uint(u_dimensions.x)), int(pNextIdx / uint(u_dimensions.x)));
  float startNext = floor(texelFetch(u_texEdgePtrOld, coordNext, 0).r + 0.5);
  
  // Safety: if pNextIdx is invalid (out of bounds logic handled by texture wrap or sentinel injection)
  // Ideally sentinel is at N. Texture size is power of 2, so N is usually < Size.
  // If N == Size, we need careful handling or larger texture.
  
  o_count = startNext - startCurrent;
}
`
    });

    // 2. Scan: Sum[i] + Sum[i-offset] -> Sum[next]
    this.progScan = createProgramSafe({
      gl,
      vertexSource: /*glsl*/`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: /*glsl*/`#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D u_texSum;
uniform int u_offset;
uniform vec2 u_dimensions;

layout(location=0) out float o_sum;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    int flatIdx = coord.y * int(u_dimensions.x) + coord.x;
    
    float self = texelFetch(u_texSum, coord, 0).r;
    
    if (flatIdx >= u_offset) {
        int neighborIdx = flatIdx - u_offset;
        ivec2 neighborCoord = ivec2(
            neighborIdx % int(u_dimensions.x), 
            neighborIdx / int(u_dimensions.x)
        );
        float other = texelFetch(u_texSum, neighborCoord, 0).r;
        o_sum = self + other;
    } else {
        o_sum = self;
    }
}`
    });

    // 3. Final: InclusiveSum - Count -> Start; Write (Start, Count)
    this.progFinal = createProgramSafe({
      gl,
      vertexSource: /*glsl*/`#version 300 es
layout(location=0) in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`,
      fragmentSource: /*glsl*/`#version 300 es
precision highp float;
precision highp usampler2D;
precision highp sampler2D;

#define encodedSortOrderWidth ${encodedSortOrderWidth}
#define particleCount ${particleCount}

uniform sampler2D u_texSum; // Inclusive Sum
uniform sampler2D u_texEdgePtrOld;      // R32F
uniform usampler2D u_order0, u_order1, u_order2, u_order3, u_order4, u_order5, u_order6, u_order7;
uniform int u_sortOffset;
uniform vec2 u_dimensions;

layout(location=0) out float o_edgePtr;

uint fetchPackedFromOrder(int texIdx, ivec2 coord, int comp) {
    uvec4 ord;
    if (texIdx == 0) ord = texelFetch(u_order0, coord, 0);
    else if (texIdx == 1) ord = texelFetch(u_order1, coord, 0);
    else if (texIdx == 2) ord = texelFetch(u_order2, coord, 0);
    else if (texIdx == 3) ord = texelFetch(u_order3, coord, 0);
    else if (texIdx == 4) ord = texelFetch(u_order4, coord, 0);
    else if (texIdx == 5) ord = texelFetch(u_order5, coord, 0);
    else if (texIdx == 6) ord = texelFetch(u_order6, coord, 0);
    else ord = texelFetch(u_order7, coord, 0);
    if (comp == 0) return ord.x; else if (comp == 1) return ord.y; else if (comp == 2) return ord.z; return ord.w;
}

uint getOldID(uint pNew) {
    int idx = int(pNew);
    int offset = u_sortOffset;
    int relIdx = idx - offset;

    if (relIdx < 0 || relIdx >= (particleCount - offset) / 128 * 128) {
        return uint(idx);
    }

    int chunkIdx = relIdx / 128; // 128 items per chunk
    int sub = relIdx % 128;
    
    int texIdx = sub / 16; // 16 items per texture (128/8)
    int rem = sub % 16;
    int comp = rem / 4;    // 4 items per component
    int byteIdx = rem % 4; // 1 byte per item

    ivec2 coord = ivec2(chunkIdx % encodedSortOrderWidth, chunkIdx / encodedSortOrderWidth);
    uint val = fetchPackedFromOrder(texIdx, coord, comp);
    uint rel = (val >> (uint(byteIdx) * 8u)) & 0xFFu;
    
    return uint(chunkIdx * 128 + offset) + rel;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    int flatIdx = coord.y * int(u_dimensions.x) + coord.x;

    float inclusiveSum = texelFetch(u_texSum, coord, 0).r;

    float count = 0.0;
    if (flatIdx < int(particleCount)) {
        // We must re-calculate count to find 'start = inclusiveSum - count'
        uint pOld = getOldID(uint(flatIdx));
        ivec2 coordOld = ivec2(int(pOld % uint(u_dimensions.x)), int(pOld / uint(u_dimensions.x)));
        float startCurrent = floor(texelFetch(u_texEdgePtrOld, coordOld, 0).r + 0.5);
        uint pNextIdx = pOld + 1u;
        ivec2 coordNext = ivec2(int(pNextIdx % uint(u_dimensions.x)), int(pNextIdx / uint(u_dimensions.x)));
        float startNext = floor(texelFetch(u_texEdgePtrOld, coordNext, 0).r + 0.5);
        count = startNext - startCurrent;
    }
    
    // Exclusive Start = Inclusive - Count
    float start = floor(inclusiveSum + 0.5) - count;
    
    o_edgePtr = start;
}`
    });

    // -- State --
    this.initMetaLoc = gl.getUniformLocation(this.progInit, 'u_inIdMassTintEdgePtr');
    this.initEdgePtrLoc = gl.getUniformLocation(this.progInit, 'u_texEdgePtrOld');
    this.initSortLocs = [];
    for (let i = 0; i < 8; i++) this.initSortLocs.push(gl.getUniformLocation(this.progInit, `u_order${i}`));
    this.initSortOffsetLoc = gl.getUniformLocation(this.progInit, 'u_sortOffset');
    this.initDimsLoc = gl.getUniformLocation(this.progInit, 'u_dimensions');

    this.texLocScan = gl.getUniformLocation(this.progScan, 'u_texSum');
    this.offLocScan = gl.getUniformLocation(this.progScan, 'u_offset');
    this.dimLocScan = gl.getUniformLocation(this.progScan, 'u_dimensions');

    this.sumLocFinal = gl.getUniformLocation(this.progFinal, 'u_texSum');
    this.finalEdgePtrLoc = gl.getUniformLocation(this.progFinal, 'u_texEdgePtrOld');
    this.finalSortLocs = [];
    for (let i = 0; i < 8; i++) this.finalSortLocs.push(gl.getUniformLocation(this.progFinal, `u_order${i}`));
    this.finalSortOffsetLoc = gl.getUniformLocation(this.progFinal, 'u_sortOffset');
    this.dimLocFinal = gl.getUniformLocation(this.progFinal, 'u_dimensions');
  }

  _createTexture() {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.width, this.height, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return tex;
  }

  /**
   * @param {{
   *   inIdMassTintEdgePtr: WebGLTexture,
   *   texEdgePtrOld: WebGLTexture,
   *   texSortOrder: WebGLTexture[],
   *   texOutput: WebGLTexture,
   *   sortOffset: number,
   *   quadVAO: WebGLVertexArrayObject
   * }} input
   */
  run({ inIdMassTintEdgePtr, texEdgePtrOld, texSortOrder, texOutput, sortOffset, quadVAO }) {
    const gl = this.gl;
    const w = this.width;
    const h = this.height;
    const N = w * h;

    gl.viewport(0, 0, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.bindVertexArray(quadVAO);

    // -- Step 1: Init --
    // Read Old Pointer via Sort Map, Compute Count = Start[Old+1] - Start[Old]
    gl.useProgram(this.progInit);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inIdMassTintEdgePtr);
    gl.uniform1i(this.initMetaLoc, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texEdgePtrOld);
    gl.uniform1i(this.initEdgePtrLoc, 1);

    if (texSortOrder) {
      for (let i = 0; i < 8; i++) {
        gl.activeTexture(gl.TEXTURE2 + i);
        gl.bindTexture(gl.TEXTURE_2D, texSortOrder[i]);
        gl.uniform1i(this.initSortLocs[i], 2 + i);
      }
    }
    
    gl.uniform1i(this.initSortOffsetLoc, sortOffset);
    gl.uniform2f(this.initDimsLoc, w, h);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texSum[0], 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // -- Step 2: Scan Loop --
    gl.useProgram(this.progScan);
    gl.uniform2f(this.dimLocScan, w, h);

    let readIdx = 0;
    let writeIdx = 1;
    const steps = Math.ceil(Math.log2(N));

    for (let i = 0; i < steps; i++) {
      const offset = 1 << i;

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texSum[readIdx]);
      gl.uniform1i(this.texLocScan, 0);
      gl.uniform1i(this.offLocScan, offset);

      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texSum[writeIdx], 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Swap
      readIdx = 1 - readIdx;
      writeIdx = 1 - writeIdx;
    }

    // Now texSum[readIdx] holds the Inclusive Prefix Sum

    // -- Step 3: Final Write --
    // Write to texOutput (R32UI)
    // Needs (InclusiveSum - Count) -> Start Pointer

    gl.useProgram(this.progFinal);
    gl.uniform2f(this.dimLocFinal, w, h);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texSum[readIdx]);
    gl.uniform1i(this.sumLocFinal, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texEdgePtrOld);
    gl.uniform1i(this.finalEdgePtrLoc, 1);

    if (texSortOrder) {
      for (let i = 0; i < 8; i++) {
        gl.activeTexture(gl.TEXTURE2 + i);
        gl.bindTexture(gl.TEXTURE_2D, texSortOrder[i]);
        gl.uniform1i(this.finalSortLocs[i], 2 + i);
      }
    }

    gl.uniform1i(this.finalSortOffsetLoc, sortOffset);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texOutput, 0);
    // Note: texOutput MUST be R32UI

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Cleanup
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);

    this.renderCount++;
  }

  /**
   * @param {{ pixels?: boolean }} [options]
   */
  valueOf({ pixels } = {}) {
    return {
      particleCount: this.particleCount,
      width: this.width,
      height: this.height,
      renderCount: this.renderCount,
      toString: () => this.toString()
    };
  }

  toString() {
    return `KEdgePrefixSum(${this.particleCount}) #${this.renderCount}`;
  }

  dispose() {
    const gl = this.gl;
    gl.deleteTexture(this.texSum[0]);
    gl.deleteTexture(this.texSum[1]);
    gl.deleteFramebuffer(this.fbo);
    gl.deleteProgram(this.progInit);
    gl.deleteProgram(this.progScan);
    gl.deleteProgram(this.progFinal);
  }
}

