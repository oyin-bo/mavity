/**
 * @param {{
 *  gl: WebGL2RenderingContext,
 *  source: string,
 *  type: number
 * }} _
 */
export function createShaderSafe({ gl, source, type }) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    const typeName = type === gl.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT';
    console.error(`Shader Error [${typeName}]:`, info);
    console.groupCollapsed('Shader Source');
    console.log(source);
    console.groupEnd();
    throw new Error('Shader compilation failed: ' + info);
  }
  return shader;
}

/**
 * @param {{
 *  gl: WebGL2RenderingContext,
 *  vertexSource: string,
 *  fragmentSource: string
 * }} _
 */
export function createProgramSafe({ gl, vertexSource, fragmentSource }) {
  const vertexShader = createShaderSafe({ gl, type: gl.VERTEX_SHADER, source: vertexSource });
  const fragmentShader = createShaderSafe({ gl, type: gl.FRAGMENT_SHADER, source: fragmentSource });
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program) || 'Unknown error linking program.');
  return program;
}

/**
 * Checks for WebGL errors and Framebuffer completeness.
 * @param {WebGL2RenderingContext} gl
 * @param {string} label - trace label to identify the call site
 */
export function glValidate(gl, label = 'Check') {
  const err = gl.getError();
  if (err !== gl.NO_ERROR) {
    let errName = 'UNKNOWN_ERROR';
    const context = /** @type {any} */ (gl);
    for (const key in context) {
      if (context[key] === err) {
        errName = key;
        break;
      }
    }
    throw new Error('WebGL Error at [' + label + ']: ' + errName + ' (' + err + ')');
  }

  const currentFBO = gl.getParameter(gl.FRAMEBUFFER_BINDING);
  if (currentFBO) {
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      let statusName = 'UNKNOWN_STATUS';
      const context = /** @type {any} */ (gl);
      for (const key in context) {
        if (context[key] === status) {
          statusName = key;
          break;
        }
      }
      throw new Error('Framebuffer Incomplete at [' + label + ']: ' + statusName + ' (' + status + ')');
    }
  }
}

/**
 * @param {{
 *  gl: WebGL2RenderingContext,
 *  width: number,
 *  height: number
 * }} _ 
 */
export function createTextureRGBA32F({ gl, width, height }) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return t;
}

/**
 * @param {{
 *  gl: WebGL2RenderingContext,
 *  width: number,
 *  height: number
 * }} _ 
 */
export function createTextureR32F({ gl, width, height }) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return t;
}

/**
 * @param {{
 *  gl: WebGL2RenderingContext,
 *  width: number,
 *  height: number
 * }} _ 
 */
export function createTextureRGBA32UI({ gl, width, height }) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, width, height, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return t;
}

