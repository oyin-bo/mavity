// @ts-check

import {
  AdditiveBlending,
  BackSide,
  Float32BufferAttribute,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Points,
  ShaderMaterial,
  Vector2,
  Texture,
  ExternalTexture
} from 'three';

import { GPUProfiler } from './gravity/monolithic/utils/gpu-profiler.js';
import { uniform } from 'three/tsl';

/**
 * Create particle rendering mesh
 * 
 * @param {{
 *   spots?: // Array mode (CPU data)
 *    { x?: number, y?: number, z?: number, mass?: number, rgb?: number }[],
 *   get?: (spot: any, coords: { x?: number, y?: number, z?: number, mass?: number, rgb?: number }) => void,
 *   textureMode?: boolean, // Texture mode (GPU data)
 *   particleCount?: number,
 *   textures?: {
 *     position: WebGLTexture | Texture,  // RGBA32F: xyz=position, w=mass
 *     color: WebGLTexture | Texture,    // RGBA: particle color
 *     size: [number, number]                   // Texture dimensions [width, height]
 *   },
 *   fog?: number | { start?: number, gray?: number }, // Common options
 *   glsl?: { definitions?: string, vertex?: string, uniforms?: Record<string, any> },
 *   enableProfiling?: boolean,
 *   gl?: WebGL2RenderingContext // Required if enableProfiling is true
 * }} options
 */
export function massSpotMesh({ spots, textureMode, particleCount, textures, get, fog, glsl, enableProfiling, gl }) {
  
  // NEW: Texture mode branch (GPU-resident data)
  if (textureMode) {
    if (!textures || !textures.position || !particleCount) {
      throw new Error('textureMode requires: particleCount, textures.position, textures.size');
    }
    
  return createTextureBasedMesh({
    particleCount,
    positionTexture: textures.position,
    colorTexture: textures.color,
    textureSize: textures.size,
    fog,
    glsl,
      enableProfiling,
      gl
    });
  }
  
  // EXISTING: Array-based mode (CPU data, unchanged)
  return createArrayBasedMesh({
    spots: /** @type {*} */(spots),
    get,
    fog,
    glsl,
    enableProfiling,
    gl
  });
}

/**
 * @param {Pick<
 *  Parameters<typeof massSpotMesh>[0],
 *  'get' | 'fog' | 'glsl' | 'enableProfiling' | 'gl'
 * > & { spots: any[] }} _
 */
function createArrayBasedMesh({ spots, get, fog, glsl, enableProfiling, gl }) {
  const dummy = {
    index: 0,
    x: 0,
    y: 0,
    z: 0,
    mass: 0,
    rgb: 0
  };

  const positions = new Float32Array([0, 0, 0]);

  let offsetBuf = new Float32Array(spots.length * 3);
  let diameterBuf = new Float32Array(spots.length);
  let colorBuf = new Uint32Array(spots.length);

  populateBuffers();

  let geometry = new InstancedBufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('offset', new InstancedBufferAttribute(offsetBuf, 3));
  geometry.setAttribute('diameter', new InstancedBufferAttribute(diameterBuf, 1));
  geometry.setAttribute('color', new InstancedBufferAttribute(colorBuf, 1));
  geometry.instanceCount = spots.length;

  // GPU Profiler (created only if enabled)
  /** @type {GPUProfiler | null} */
  let profiler = null;
  if (enableProfiling && gl) {
    profiler = new GPUProfiler(gl);
  }

  let fogStart = 0.6;
  let fogGray = 1.0;
  if (fog) {
    if (typeof fog === 'number') {
      fogStart = fog;
      fogGray = fog * 4 / 10;
    } else {
      if (fog.start) fogStart = fog.start;
      if (fog.gray) fogGray = fog.gray;
    }
  }

  const material = new ShaderMaterial({
    uniforms: {
      fogStart: { value: fogStart },
      fogGray: { value: fogGray }
    },
    blending: AdditiveBlending,
    vertexShader: (glsl?.definitions || '') + /* glsl */`
            precision highp float;
            attribute vec3 offset;
            attribute float diameter;
            attribute uint color;
            varying float vDiameter;
            varying float vFogDist;
            varying vec4 vColor;
            void main(){
              vDiameter = diameter;
              gl_Position = projectionMatrix * (modelViewMatrix * vec4(offset, 1.0));
              vec4 viewPosition = modelViewMatrix * vec4(offset, 1.0);
              float distanceToCamera = length(viewPosition.xyz);
              float pointScaleFactor = 1600.0;
              gl_PointSize = abs(diameter) * pointScaleFactor / distanceToCamera;
              uint rInt = (color / uint(256 * 256 * 256)) % uint(256);
              uint gInt = (color / uint(256 * 256)) % uint(256);
              uint bInt = (color / uint(256)) % uint(256);
              uint aInt = (color) % uint(256);
              vColor = vec4(float(rInt) / 255.0f, float(gInt) / 255.0f, float(bInt) / 255.0f, float(aInt) / 255.0f);
              vFogDist = distance(cameraPosition, offset);
              ${glsl?.vertex || ''}
            }
          `,
    fragmentShader: /* glsl */`
            precision highp float;
            varying vec4 vColor;
            varying float vFogDist;
            varying float vDiameter;
            uniform float fogStart;
            uniform float fogGray;
            void main() {
              gl_FragColor = vColor;
              float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
              dist = vDiameter < 0.0 ? dist * 2.0 : dist;
              float rad = 0.25;
              float areola = rad * 2.0;
              float bodyRatio =
                dist < rad ? 1.0 :
                dist > areola ? 0.0 :
                (areola - dist) / (areola - rad);
              float radiusRatio =
                dist < 0.5 ? 1.0 - dist * 2.0 : 0.0;
              float fogRatio = vFogDist < fogStart ? 0.0 : vFogDist > fogGray ? 1.0 : (vFogDist - fogStart) / (fogGray - fogStart);
              vec4 tintColor = vColor;
              tintColor.a = radiusRatio;
              gl_FragColor = mix(gl_FragColor, vec4(1.0,1.0,1.0,0.7), fogRatio * 0.7);
              gl_FragColor = vDiameter < 0.0 ? vec4(0.6,0.0,0.0,1.0) : gl_FragColor;
              gl_FragColor.a = bodyRatio;
            }
          `,
    side: BackSide,
    forceSinglePass: true,
    transparent: true,
    depthWrite: false
  });

  const mesh = new Points(geometry, material);
  
  // Add profiling support with onBeforeRender/onAfterRender
  if (profiler) {
    mesh.onBeforeRender = () => {
      profiler.update();  // Auto-update profiler state each frame
      profiler.begin('particle_draw');
    };
    mesh.onAfterRender = () => {
      profiler.end();
    };
  }
  
  // Return wrapper object instead of polluting Three.js Points
  return {
    mesh,
    update: updateSpots,
    stats: getProfilingStats
  };

  function populateBuffers() {
    for (let i = 0; i < spots.length; i++) {
      const spot = spots[i];
      dummy.index = i;
      dummy.x = spot.x || 0;
      dummy.y = spot.y || 0;
      dummy.z = spot.z || 0;
      dummy.mass = spot.mass || 0;
      dummy.rgb = spot.rgb || 0;
      if (typeof get === 'function') get(spot, dummy);
      offsetBuf[i * 3 + 0] = dummy.x;
      offsetBuf[i * 3 + 1] = dummy.y;
      offsetBuf[i * 3 + 2] = dummy.z;
      diameterBuf[i] = dummy.mass;
      colorBuf[i] = dummy.rgb << 8;
    }
  }

  /**
   * @param {any[]} newSpots
   */
  function updateSpots(newSpots) {
    spots = newSpots;
    if (newSpots.length > geometry.instanceCount || newSpots.length < geometry.instanceCount / 2) {
      const newAllocateCount = Math.max(Math.floor(newSpots.length * 1.5), newSpots.length + 300);
      offsetBuf = new Float32Array(newAllocateCount * 3);
      diameterBuf = new Float32Array(newAllocateCount);
      colorBuf = new Uint32Array(newAllocateCount);
      populateBuffers();
      const oldGeometry = geometry;
      geometry = new InstancedBufferGeometry();
      geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
      geometry.setAttribute('offset', new InstancedBufferAttribute(offsetBuf, 3));
      geometry.setAttribute('diameter', new InstancedBufferAttribute(diameterBuf, 1));
      geometry.setAttribute('color', new InstancedBufferAttribute(colorBuf, 1));
      geometry.instanceCount = newAllocateCount;
      mesh.geometry = geometry;
      oldGeometry.dispose();
    } else {
      populateBuffers();
      geometry.attributes['offset'].needsUpdate = true;
      geometry.attributes['diameter'].needsUpdate = true;
      geometry.attributes['color'].needsUpdate = true;
    }
  }

  /**
   * Get profiling statistics
   * @returns {{ particle_draw?: number } | null}
   */
  function getProfilingStats() {
    if (!profiler || !profiler.enabled) return null;
    return profiler.getAll();
  }
}

/**
 * Internal: Create texture-based particle mesh (GPU data flow)
 * Accepts raw WebGLTexture objects for zero-copy GPU pipeline
 * @param {{
 *  particleCount: number,
 *  positionTexture: WebGLTexture | Texture,
 *  colorTexture: WebGLTexture | Texture,
 *  textureSize: [number, number],
 *  fog?: number | { start?: number, gray?: number },
 *  glsl?: { definitions?: string, vertex?: string, uniforms?: Record<string, any> },
 *  enableProfiling?: boolean,
 *  gl?: WebGL2RenderingContext
 * }} _
 */
function createTextureBasedMesh({ particleCount, positionTexture, colorTexture, textureSize, fog, glsl, enableProfiling, gl }) {
  const dummyPositions = new Float32Array([0, 0, 0]);
  let geometry = new InstancedBufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(dummyPositions, 3));
  geometry.instanceCount = particleCount;

  let fogStart = 0.6, fogGray = 1.0;
  if (typeof fog === 'number') {
    fogStart = fog;
    fogGray = fog * 4 / 10;
  } else if (fog) {
    if (fog.start) fogStart = fog.start;
    if (fog.gray) fogGray = fog.gray;
  }

  // Wrap raw WebGLTexture in THREE.ExternalTexture
  /**
   * Wraps a raw WebGLTexture or THREE.Texture for use in the shader.
   * @param {WebGLTexture | Texture} tex
   */
  const wrapTexture = (tex) => {
    if (!tex) return null;
    // If it's already a THREE.Texture, use it
    if (/** @type {Texture} */ (tex).isTexture) return tex;
    // Wrap raw WebGLTexture in ExternalTexture for zero-copy GPU pipeline
    return new ExternalTexture(tex);
  };

  const material = new ShaderMaterial({
    uniforms: {
      fogStart: { value: fogStart },
      fogGray: { value: fogGray },
      u_positionTexture: { value: wrapTexture(positionTexture) },
      u_colorTexture: { value: wrapTexture(colorTexture) },
      u_texSize: { value: new Vector2(textureSize[0], textureSize[1]) },
      ...(
        !glsl?.uniforms ? {} :
          Object.fromEntries(
            Object.entries(glsl.uniforms)
              .map(([k, v]) => [k, v.value ? v : { value: v }]))
      )
    },
    blending: AdditiveBlending,
    vertexShader: (glsl?.definitions || '') + /* glsl */`
      precision highp float;

      uniform sampler2D u_positionTexture;
      uniform sampler2D u_colorTexture;
      uniform vec2 u_texSize;

      varying float vDiameter;
      varying float vFogDist;
      varying vec4 vColor;

      ivec2 indexToTexCoord(int index, vec2 texSize) {
        int w = int(texSize.x);
        return ivec2(index % w, index / w);
      }

      void main() {
        ivec2 texCoord = indexToTexCoord(gl_InstanceID, u_texSize);
        vec4 posData = texelFetch(u_positionTexture, texCoord, 0);
        vec4 colorData = texelFetch(u_colorTexture, texCoord, 0);

        vec3 offset = posData.xyz;
        float diameter = posData.w * 0.015;  // mass to size

        vDiameter = diameter;
        gl_Position = projectionMatrix * (modelViewMatrix * vec4(offset, 1.0));

        vec4 viewPosition = modelViewMatrix * vec4(offset, 1.0);
        float distanceToCamera = length(viewPosition.xyz);
        float pointScaleFactor = 1600.0;
        gl_PointSize = abs(diameter) * pointScaleFactor / distanceToCamera;
        
        vColor = colorData;
        vFogDist = distance(cameraPosition, offset);
        
        ${glsl?.vertex || ''}
      }
    `,
    fragmentShader: /* glsl */`
      precision highp float;
      
      varying vec4 vColor;
      varying float vFogDist;
      varying float vDiameter;
      
      uniform float fogStart;
      uniform float fogGray;
      
      void main() {
        gl_FragColor = vColor;
        float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
        dist = vDiameter < 0.0 ? dist * 2.0 : dist;
        float rad = 0.25;
        float areola = rad * 2.0;
        float bodyRatio =
          dist < rad ? 1.0 :
          dist > areola ? 0.0 :
          (areola - dist) / (areola - rad);
        float radiusRatio =
          dist < 0.5 ? 1.0 - dist * 2.0 : 0.0;
        float fogRatio = vFogDist < fogStart ? 0.0 : vFogDist > fogGray ? 1.0 : (vFogDist - fogStart) / (fogGray - fogStart);
        vec4 tintColor = vColor;
        tintColor.a = radiusRatio;
        gl_FragColor = mix(gl_FragColor, vec4(1.0,1.0,1.0,0.7), fogRatio * 0.7);
        gl_FragColor = vDiameter < 0.0 ? vec4(0.6,0.0,0.0,1.0) : gl_FragColor;
        gl_FragColor.a = bodyRatio;
      }
    `,
    side: BackSide,
    forceSinglePass: true,
    transparent: true,
    depthWrite: false
  });

  const mesh = new Points(geometry, material);
  
  // GPU Profiler (created only if enabled)
  /** @type {GPUProfiler | null} */
  let profiler = null;
  if (enableProfiling && gl) {
    profiler = new GPUProfiler(gl);
  }
  
  // Add profiling support with onBeforeRender/onAfterRender
  if (profiler) {
    mesh.onBeforeRender = () => {
      profiler.update();  // Auto-update profiler state each frame
      profiler.begin('particle_draw');
    };
    mesh.onAfterRender = () => {
      profiler.end();
    };
  }

  // Return wrapper object instead of polluting Three.js Points
  return {
    mesh,
    update: updateTextures,
    stats: getProfilingStats
  };

  /**
   * Updates the position and color textures.
   * @param {WebGLTexture | Texture} newPositionTexture 
   * @param {WebGLTexture | Texture} newColorTexture 
   * @param {Record<string, any>} [uniforms]
   */
  function updateTextures(newPositionTexture, newColorTexture, uniforms) {
    material.uniforms.u_positionTexture.value = wrapTexture(newPositionTexture);
    if (newColorTexture) {
      material.uniforms.u_colorTexture.value = wrapTexture(newColorTexture);
    }

    if (uniforms) {
      for (const key in uniforms) {
        if (material.uniforms[key]) {
          material.uniforms[key].value = uniforms[key];
        }
      }
    }
  }

  /**
   * Get profiling statistics
   * @returns {{ particle_draw?: number } | null}
   */
  function getProfilingStats() {
    if (!profiler || !profiler.enabled) return null;
    return profiler.getAll();
  }

}
