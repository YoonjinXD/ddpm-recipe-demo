// 28x28 images as 3D bar fields (three.js, instanced meshes).
//   bar height  = (pixel + 1) * HS        (black background = flat floor, white = tall)
//   red sleeve  = the part of a bar that is noise (between v and v - noisePart)
//   amber arrow = where a pixel is about to move (v -> v + delta)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const N = 28, P = N * N;
export const HS = 1.5;
const HMAX = 7.5;
const GAP = 9;
const COL_NOISE = 0xe66767, COL_FLOW = 0xeda100, COL_OVER = 0xeda100, COL_BAD = 0x8a2b2b;

const H = v => (v + 1) * HS;
const clampH = h => (h > HMAX ? HMAX : h < -HMAX ? -HMAX : h);
const ease = k => (k <= 0 ? 0 : k >= 1 ? 1 : 1 - Math.pow(1 - k, 3));

const _m = new THREE.Matrix4(), _p = new THREE.Vector3(), _s = new THREE.Vector3(), _c = new THREE.Color();
const Q_UP = new THREE.Quaternion(), Q_DOWN = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
const ZERO = new Float32Array(P);

class BarGrid {
  constructor(scene, cx) {
    this.cx = cx;
    const box = new THREE.BoxGeometry(1, 1, 1);
    this.bars = new THREE.InstancedMesh(box, new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.0 }), P);
    // noise part = an open sleeve (side faces only): top faces stay gray, red shows as a band whose height is the noise
    const sleeve = new THREE.BoxGeometry(1, 1, 1);
    const idx = sleeve.index.array;
    sleeve.setIndex([...idx.slice(0, 12), ...idx.slice(24, 36)]);
    sleeve.clearGroups();
    this.ghost = new THREE.InstancedMesh(sleeve, new THREE.MeshStandardMaterial({
      color: COL_NOISE, emissive: 0x6a1a1a, transparent: true, opacity: 0.8, roughness: 0.5, depthWrite: false,
      side: THREE.DoubleSide }), P);
    this.ghost.renderOrder = 2;
    const arrMat = new THREE.MeshStandardMaterial({ color: COL_FLOW, emissive: 0x4a3000, roughness: 0.45, depthTest: false });
    this.shaft = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.11, 0.11, 1, 6), arrMat, P);
    this.head = new THREE.InstancedMesh(new THREE.ConeGeometry(0.34, 1, 10), arrMat, P);
    this.shaft.renderOrder = this.head.renderOrder = 3;
    for (const m of [this.bars, this.ghost, this.shaft, this.head]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.frustumCulled = false;
      scene.add(m);
    }
    for (let i = 0; i < P; i++) this.bars.setColorAt(i, _c.setRGB(0.3, 0.3, 0.3));
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(N + 1.2, N + 1.2),
      new THREE.MeshBasicMaterial({ color: 0x141413, transparent: true, opacity: 0.9, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(cx, -0.02, 0);
    scene.add(floor);
    const grid = new THREE.GridHelper(N, N, 0x2c2c2a, 0x262624);
    grid.position.set(cx, 0.0, 0);
    scene.add(grid);
    this.cur = { v: new Float32Array(P).fill(-1), g: new Float32Array(P), a: new Float32Array(P) };
    this.from = { v: new Float32Array(P).fill(-1), g: new Float32Array(P), a: new Float32Array(P) };
    this.to = { v: new Float32Array(P).fill(-1), g: new Float32Array(P), a: new Float32Array(P) };
    this.t0 = -1; this.dur = 420; this.anim = true;
  }

  // print each pixel's value on the top face of its bar (one texture for the whole grid)
  showValues(values) {
    const cell = 64, cv = document.createElement('canvas');
    cv.width = cv.height = N * cell;
    const ctx = cv.getContext('2d');
    ctx.font = `600 ${cell * 0.34}px system-ui, -apple-system, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#fff';
    const fmt = v => (Math.abs(v + 1) < 0.005 ? '−1' : v.toFixed(2).replace(/^(-?)0\./, '$1.').replace('-', '−'));
    for (let i = 0; i < P; i++) ctx.fillText(fmt(values[i]), ((i % N) + 0.5) * cell, (((i / N) | 0) + 0.5) * cell);
    const tex = new THREE.CanvasTexture(cv);
    tex.anisotropy = 8;

    const mat = this.bars.material;
    mat.onBeforeCompile = shader => {
      shader.uniforms.numTex = { value: tex };
      shader.vertexShader = 'varying vec2 vCell;\nvarying float vTop;\n' + shader.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
        vCell = vec2(float(gl_InstanceID % ${N}) + position.x + 0.5, float(gl_InstanceID / ${N}) + position.z + 0.5);
        vTop = step(0.5, normal.y);`);
      shader.fragmentShader = 'uniform sampler2D numTex;\nvarying vec2 vCell;\nvarying float vTop;\n' + shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
        if (vTop > 0.5) {
          float ink = texture2D(numTex, vec2(vCell.x / ${N}.0, 1.0 - vCell.y / ${N}.0)).a;
          float lum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
          diffuseColor.rgb = mix(diffuseColor.rgb, lum > 0.45 ? vec3(0.06) : vec3(0.9), ink);
        }`);
    };
    mat.needsUpdate = true;
  }

  set({ v, ghost = null, arrows = null }, instant = false) {
    for (const k of ['v', 'g', 'a']) this.from[k].set(this.cur[k]);
    this.to.v.set(v);
    this.to.g.set(ghost || ZERO);
    this.to.a.set(arrows || ZERO);
    this.t0 = instant ? -1e9 : performance.now();
    this.anim = true;
  }

  tick(now) {
    if (!this.anim) return false;
    const k = ease((now - this.t0) / this.dur);
    for (const key of ['v', 'g', 'a']) {
      const f = this.from[key], t = this.to[key], c = this.cur[key];
      for (let i = 0; i < P; i++) {
        const a = f[i], b = t[i];
        c[i] = Number.isFinite(a) && Number.isFinite(b) ? a + (b - a) * k : b;
      }
    }
    this.write();
    if (k >= 1) this.anim = false;
    return true;
  }

  write() {
    const { v, g, a } = this.cur;
    for (let i = 0; i < P; i++) {
      const r = (i / N) | 0, c = i % N;
      const x = this.cx + (c - 13.5), z = r - 13.5;
      const val = v[i];
      // ---- bar
      let h, over = false, bad = !Number.isFinite(val);
      if (bad) h = 0.12;
      else { h = H(val); const hc = clampH(h); over = hc !== h; h = hc; }
      const hh = Math.abs(h) < 0.03 ? 0.03 : Math.abs(h);
      _m.makeScale(0.84, hh, 0.84); _m.setPosition(x, (h >= 0 ? hh : -hh) / 2, z);
      this.bars.setMatrixAt(i, _m);
      if (bad) _c.setHex(COL_BAD);
      else if (over) _c.setHex(COL_OVER);
      else { const L = 0.16 + 0.8 * Math.min(1, Math.max(0, (val + 1) / 2)); _c.setRGB(L, L * 0.995, L * 0.98); }
      this.bars.setColorAt(i, _c);
      // ---- noise sleeve
      const n = g[i];
      if (!bad && Number.isFinite(n) && Math.abs(n) * HS > 0.02) {
        const y1 = clampH(H(val)), y2 = clampH(H(val - n));
        const len = Math.max(Math.abs(y1 - y2), 0.02);
        _m.makeScale(0.94, len, 0.94); _m.setPosition(x, (y1 + y2) / 2, z);
      } else { _m.makeScale(0, 0, 0); }
      this.ghost.setMatrixAt(i, _m);
      // ---- arrow
      const d = a[i];
      const y1 = bad ? 0 : clampH(H(val)), y2 = bad ? 0 : clampH(H(val + d));
      const L = Math.abs(y2 - y1);
      if (!bad && Number.isFinite(d) && L > 0.22) {
        const dir = y2 > y1 ? 1 : -1;
        const hl = Math.min(0.75, L * 0.45), sl = L - hl;
        _m.compose(_p.set(x, y1 + dir * sl / 2, z), Q_UP, _s.set(1, sl, 1));
        this.shaft.setMatrixAt(i, _m);
        _m.compose(_p.set(x, y1 + dir * (sl + hl / 2), z), dir > 0 ? Q_UP : Q_DOWN, _s.set(1, hl, 1));
        this.head.setMatrixAt(i, _m);
      } else {
        _m.makeScale(0, 0, 0);
        this.shaft.setMatrixAt(i, _m);
        this.head.setMatrixAt(i, _m);
      }
    }
    for (const m of [this.bars, this.ghost, this.shaft, this.head]) m.instanceMatrix.needsUpdate = true;
    this.bars.instanceColor.needsUpdate = true;
  }
}

export class BarScene {
  constructor(el, { grids = 1, elev = 0.62, azim = 0, zoom = false } = {}) {
    this.el = el;
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.prepend(r.domElement);
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(26, 1, 1, 4000);
    this.span = grids * N + (grids - 1) * GAP;
    this.dir = new THREE.Vector3(Math.sin(azim) * Math.cos(elev), Math.sin(elev), Math.cos(azim) * Math.cos(elev));
    this.target = new THREE.Vector3(0, 2.2, 1.5);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x202020, 1.35));
    const sun = new THREE.DirectionalLight(0xffffff, 1.9);
    sun.position.set(-30, 60, 40);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(40, 20, -30);
    this.scene.add(fill);
    this.ctl = new OrbitControls(this.cam, r.domElement);
    Object.assign(this.ctl, { enableDamping: true, dampingFactor: 0.12, enableZoom: zoom, minDistance: 20, enablePan: false,
      minPolarAngle: 0.12, maxPolarAngle: 1.5, rotateSpeed: 0.7 });
    this.ctl.target.copy(this.target);
    this.ctl.addEventListener('change', () => { this.dirty = true; });
    this.grids = [];
    for (let k = 0; k < grids; k++) this.grids.push(new BarGrid(this.scene, -this.span / 2 + N / 2 + k * (N + GAP)));
    this.labelBox = el.querySelector('.labels3d') || el.appendChild(Object.assign(document.createElement('div'), { className: 'labels3d' }));
    this.labelEls = this.grids.map(() => this.labelBox.appendChild(Object.assign(document.createElement('div'), { className: 'lab3d' })));
    this.legendEl = el.appendChild(Object.assign(document.createElement('div'), { className: 'legend3d' }));
    el.appendChild(Object.assign(document.createElement('div'), { className: 'hint3d', textContent: 'drag to rotate · double-click to reset' }));
    this.visible = true;
    this.fit();
    new ResizeObserver(() => this.fit()).observe(el);
    new IntersectionObserver(es => { this.visible = es[0].isIntersecting; if (this.visible) this.dirty = true; }).observe(el);
    const loop = now => { requestAnimationFrame(loop); this.frame(now); };
    requestAnimationFrame(loop);
    // double-click resets the view
    r.domElement.addEventListener('dblclick', () => this.resetView());
  }

  fit() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.cam.aspect = w / h;
    const vfov = THREE.MathUtils.degToRad(this.cam.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.cam.aspect);
    const dW = (this.span / 2 + 2) / Math.tan(hfov / 2);
    const dH = 15 / Math.tan(vfov / 2);
    this.dist = Math.max(dW, dH) + 4;
    const cur = this.cam.position.clone().sub(this.ctl.target);
    const dir = this.fitted ? cur.normalize() : this.dir.clone();
    this.fitted = true;
    this.cam.position.copy(this.ctl.target).addScaledVector(dir, this.dist);
    this.cam.updateProjectionMatrix();
    this.dirty = true;
  }

  resetView() {
    this.ctl.target.copy(this.target);
    this.cam.position.copy(this.target).addScaledVector(this.dir, this.dist);
    this.dirty = true;
  }

  setLegend(items) { // [[color, label], ...]
    this.legendEl.innerHTML = items.map(([c, l]) => `<span><i class="sw" style="background:${c}"></i>${l}</span>`).join('');
  }

  setLabels(htmls) { htmls.forEach((h, k) => { if (this.labelEls[k]) this.labelEls[k].innerHTML = h; }); this.dirty = true; }

  set(k, spec, instant = false) { this.grids[k].set(spec, instant); }

  showValues(k, values) { this.grids[k].showValues(values); this.dirty = true; }

  frame(now) {
    if (!this.visible) return;
    const moved = this.ctl.update();
    let anim = false;
    for (const g of this.grids) anim = g.tick(now) || anim;
    if (moved || anim || this.dirty) {
      this.renderer.render(this.scene, this.cam);
      this.placeLabels();
      this.dirty = false;
    }
  }

  placeLabels() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.grids.forEach((g, k) => {
      const p = new THREE.Vector3(g.cx, 0, 15.2).project(this.cam);
      const el = this.labelEls[k];
      el.style.left = `${(p.x * 0.5 + 0.5) * w}px`;
      el.style.top = `${Math.min(h - 46, (-p.y * 0.5 + 0.5) * h + 8)}px`;
    });
  }
}

// 2D thumbnail of a 28x28 field (NaN -> dark red)
export function drawThumb(canvas, v) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(N, N);
  for (let i = 0; i < P; i++) {
    const x = v[i];
    let r, g, b;
    if (!Number.isFinite(x)) { r = 150; g = 40; b = 40; }
    else { const q = Math.max(0, Math.min(255, (x + 1) * 127.5)); r = g = b = q; }
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
