// @ts-check
import { createProgramSafe } from './utils.js';

/**
 * KEdgeRelocation
 * 
 * Relocates edges from the old storage layout to the new storage layout
 * based on the Particle Reshuffling (Sort Order).
 * 
 * Pipeline:
 * 1. For each target edge slot E_new in `texEdgeStoreNext`.
 * 2. Identify owner P_new using `texCoarseMap` (Accel) + Linear Scan/Bisection.
 * 3. Calculate local edge index `L = E_new - Start_new[P_new]`.
 * 4. Identify original particle P_old using `texSortOrder`.
 * 5. Retrieve original start `Start_old = texEdgePtrOld[P_old].start`.
 * 6. Calculate source edge index `E_old = Start_old + L`.
 * 7. Read old target stored at `texEdgeStoreOld[E_old]`.
 * 8. Translate old target physical index -> PID -> new physical index.
 * 9. Write to `texEdgeStoreNext`.
 */
export class KEdgeRelocation {
  /**
   * @param {{
   *   gl: WebGL2RenderingContext,
   *   edgeStoreWidth: number,
   *   edgeStoreHeight: number,
   *   coarseWidth: number,
   *   coarseHeight: number,
   *   particleDataWidth: number,
   *   particleDataHeight: number,
   *   encodedSortOrderWidth: number,
   *   startStride: number,
   *   edgeCount: number,
   *   particleCount: number
   * }} options
   */
  constructor({
    gl,
    edgeStoreWidth, edgeStoreHeight,
    coarseWidth, coarseHeight,
    particleDataWidth, particleDataHeight,
    encodedSortOrderWidth,
    startStride,
    edgeCount,
    particleCount
  }) {
    this.gl = gl;
    this.edgeStoreWidth = edgeStoreWidth;
    this.edgeStoreHeight = edgeStoreHeight;
    this.coarseWidth = coarseWidth;
    this.coarseHeight = coarseHeight;
    this.particleDataWidth = particleDataWidth;
    this.particleDataHeight = particleDataHeight;
    this.encodedSortOrderWidth = encodedSortOrderWidth;
    this.stride = startStride || 128;
    this.edgeCount = edgeCount;

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
precision highp isampler2D;
precision highp sampler2D;

#define encodedSortOrderWidth ${encodedSortOrderWidth}
#define edgeStoreWidth ${edgeStoreWidth}
#define edgeCount ${edgeCount}
#define particleCount ${particleCount}
#define particleDataWidth ${particleDataWidth}
#define particleDataHeight ${particleDataHeight}

uniform sampler2D u_texEdgeStoreOld;
uniform sampler2D u_texCoarseMap;
uniform sampler2D u_texEdgePtrNew;
uniform sampler2D u_texEdgePtrOld;

uniform usampler2D u_order0, u_order1, u_order2, u_order3, u_order4, u_order5, u_order6, u_order7;

uniform sampler2D u_texIdMassTintEdgePtrOld; // RGBA32F: x=PID
uniform sampler2D u_texIdentityNew; // R32F: PID->NewPhys

uniform vec2 u_slotsRes; // Resolution of Identity Map

uniform int u_stride;
uniform int u_coarseWidth;
uniform int u_sortOffset;

layout(location=0) out float o_edgeValue;

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
    int flatIdx = coord.y * int(edgeStoreWidth) + coord.x;

    uint e_new = uint(flatIdx);
    if (e_new >= uint(edgeCount)) {
        o_edgeValue = -1.0;
        return;
    }

    // 1. Find P_new using Coarse Map
    int coarseIdx = flatIdx / u_stride;
    ivec2 coarseCoord = ivec2(
        coarseIdx % u_coarseWidth,
        coarseIdx / u_coarseWidth
    );
    uint p_new = uint(floor(texelFetch(u_texCoarseMap, coarseCoord, 0).r + 0.5));

    // 2. Refine P_new
    for (int k=0; k<256; k++) { // bumped to 256 for safety
        if (p_new >= uint(particleCount)) break;
        
        ivec2 pCoord = ivec2(int(p_new % uint(particleDataWidth)), int(p_new / uint(particleDataWidth)));
        uint start = uint(floor(texelFetch(u_texEdgePtrNew, pCoord, 0).r + 0.5));
        
        // Count = Ptr[i+1] - Ptr[i]
        ivec2 pCoordNext = ivec2(int((p_new + 1u) % uint(particleDataWidth)), int((p_new + 1u) / uint(particleDataWidth)));
        uint startNext = uint(floor(texelFetch(u_texEdgePtrNew, pCoordNext, 0).r + 0.5));
        uint count = startNext - start;

        if (e_new >= start && e_new < start + count) {
            break;
        }
        p_new++;
    }

    // 3. Local Index
    ivec2 pNewCoord = ivec2(int(p_new % uint(particleDataWidth)), int(p_new / uint(particleDataWidth)));
    uint start_new = uint(floor(texelFetch(u_texEdgePtrNew, pNewCoord, 0).r + 0.5));
    uint offset = e_new - start_new;

    // 4. Find P_old
    uint p_old = getOldID(p_new);

    // 5. Start_old
    ivec2 pOldCoord = ivec2(int(p_old % uint(particleDataWidth)), int(p_old / uint(particleDataWidth)));
    uint start_old = uint(floor(texelFetch(u_texEdgePtrOld, pOldCoord, 0).r + 0.5));

    // 6. E_old
    uint e_old = start_old + offset;

    // 7. Get Old Target Physical Index
    ivec2 eOldCoord = ivec2(int(e_old % uint(edgeStoreWidth)), int(e_old / uint(edgeStoreWidth)));
    uint oldTargetPhys = uint(floor(texelFetch(u_texEdgeStoreOld, eOldCoord, 0).r + 0.5));

    if (oldTargetPhys == 4294967295u) { // Handle -1 or uninitialized
        o_edgeValue = -1.0;
        return;
    }

    // 8. Translate (OldPhys -> PID -> NewPhys)
    ivec2 oldTargetCoord = ivec2(int(oldTargetPhys % uint(particleDataWidth)), int(oldTargetPhys / uint(particleDataWidth)));
    // Old Meta texture contains PID in .x (and it's a FLOAT texture, so we cast)
    float pidFloat = texelFetch(u_texIdMassTintEdgePtrOld, oldTargetCoord, 0).r;
    uint pid = uint(floor(pidFloat + 0.5));

    // Now look up new physical location for this PID in the identity map
    ivec2 identityCoord = ivec2(int(pid % uint(u_slotsRes.x)), int(pid / uint(u_slotsRes.x)));
    float newTargetPhys = texelFetch(u_texIdentityNew, identityCoord, 0).r;

    o_edgeValue = newTargetPhys;
}`
    });

    this.uEdgeStoreOldLoc = gl.getUniformLocation(this.program, 'u_texEdgeStoreOld');
    this.uCoarseMapLoc = gl.getUniformLocation(this.program, 'u_texCoarseMap');
    this.uEdgePtrNewLoc = gl.getUniformLocation(this.program, 'u_texEdgePtrNew');
    this.uEdgePtrOldLoc = gl.getUniformLocation(this.program, 'u_texEdgePtrOld');
    this.uIdMassTintEdgePtrOldLoc = gl.getUniformLocation(this.program, 'u_texIdMassTintEdgePtrOld');
    this.uIdentityNewLoc = gl.getUniformLocation(this.program, 'u_texIdentityNew');
    this.uSlotsResLoc = gl.getUniformLocation(this.program, 'u_slotsRes');

    // 8 Order Textures
    this.uOrderLocs = [];
    for (let i = 0; i < 8; i++) this.uOrderLocs.push(gl.getUniformLocation(this.program, `u_order${i}`));

    this.uStrideLoc = gl.getUniformLocation(this.program, 'u_stride');
    this.uCoarseWidthLoc = gl.getUniformLocation(this.program, 'u_coarseWidth');
    this.uSortOffsetLoc = gl.getUniformLocation(this.program, 'u_sortOffset');

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.fbo = gl.createFramebuffer();
  }

  /**
   * @param {{
   *   texEdgeStoreOld: WebGLTexture,
   *   texEdgeStoreNew: WebGLTexture,
   *   texCoarseMap: WebGLTexture,
   *   texEdgePtrNew: WebGLTexture,
   *   texEdgePtrOld: WebGLTexture,
   *   texIdMassTintEdgePtrOld: WebGLTexture,
   *   texIdentityNew: WebGLTexture,
   *   slotsRes: { x: number, y: number },
   *   texSortOrder: WebGLTexture[],
   *   sortOffset: number
   * }} input
   */
  run({ texEdgeStoreOld, texEdgeStoreNew, texCoarseMap, texEdgePtrNew, texEdgePtrOld, texIdMassTintEdgePtrOld, texIdentityNew, slotsRes, texSortOrder, sortOffset }) {
    const gl = this.gl;

    gl.viewport(0, 0, this.edgeStoreWidth, this.edgeStoreHeight);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    // Bind Textures
    let unit = 0;

    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texEdgeStoreOld); gl.uniform1i(this.uEdgeStoreOldLoc, unit++);
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texCoarseMap); gl.uniform1i(this.uCoarseMapLoc, unit++);
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texEdgePtrNew); gl.uniform1i(this.uEdgePtrNewLoc, unit++);
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texEdgePtrOld); gl.uniform1i(this.uEdgePtrOldLoc, unit++);

    for (let i = 0; i < 8; i++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texSortOrder[i]);
      gl.uniform1i(this.uOrderLocs[i], unit++);
    }

    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texIdMassTintEdgePtrOld); gl.uniform1i(this.uIdMassTintEdgePtrOldLoc, unit++);
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texIdentityNew); gl.uniform1i(this.uIdentityNewLoc, unit++);

    // Uniforms
    gl.uniform1i(this.uStrideLoc, this.stride);
    gl.uniform2f(this.uSlotsResLoc, slotsRes.x, slotsRes.y);
    gl.uniform1i(this.uCoarseWidthLoc, this.coarseWidth);
    gl.uniform1i(this.uSortOffsetLoc, sortOffset);

    // Framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texEdgeStoreNew, 0);

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


