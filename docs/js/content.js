// Text, references, and the "which paper used this combination" logic.

export const REFS = {
  sohl2015: ['Sohl-Dickstein et al., 2015', 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics', 'ICML 2015', 'https://arxiv.org/abs/1503.03585'],
  song2019: ['Song & Ermon, 2019', 'Generative Modeling by Estimating Gradients of the Data Distribution', 'NeurIPS 2019', 'https://arxiv.org/abs/1907.05600'],
  ho2020: ['Ho, Jain & Abbeel, 2020', 'Denoising Diffusion Probabilistic Models', 'NeurIPS 2020', 'https://arxiv.org/abs/2006.11239'],
  song2021ddim: ['Song, Meng & Ermon, 2021', 'Denoising Diffusion Implicit Models', 'ICLR 2021', 'https://arxiv.org/abs/2010.02502'],
  song2021sde: ['Song et al., 2021', 'Score-Based Generative Modeling through Stochastic Differential Equations', 'ICLR 2021', 'https://arxiv.org/abs/2011.13456'],
  nichol2021: ['Nichol & Dhariwal, 2021', 'Improved Denoising Diffusion Probabilistic Models', 'ICML 2021', 'https://arxiv.org/abs/2102.09672'],
  salimans2022: ['Salimans & Ho, 2022', 'Progressive Distillation for Fast Sampling of Diffusion Models', 'ICLR 2022', 'https://arxiv.org/abs/2202.00512'],
  karras2022: ['Karras et al., 2022', 'Elucidating the Design Space of Diffusion-Based Generative Models', 'NeurIPS 2022', 'https://arxiv.org/abs/2206.00364'],
  rombach2022: ['Rombach et al., 2022', 'High-Resolution Image Synthesis with Latent Diffusion Models', 'CVPR 2022', 'https://arxiv.org/abs/2112.10752'],
  ramesh2022: ['Ramesh et al., 2022', 'Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL·E 2)', 'arXiv 2022', 'https://arxiv.org/abs/2204.06125'],
  ho2022video: ['Ho et al., 2022', 'Imagen Video: High Definition Video Generation with Diffusion Models', 'arXiv 2022', 'https://arxiv.org/abs/2210.02303'],
  jabri2023: ['Jabri, Fleet & Chen, 2023', 'Scalable Adaptive Computation for Iterative Generation (sigmoid schedule)', 'ICML 2023', 'https://arxiv.org/abs/2212.11972'],
  lin2024: ['Lin et al., 2024', 'Common Diffusion Noise Schedules and Sample Steps Are Flawed', 'WACV 2024', 'https://arxiv.org/abs/2305.08891'],
  lipman2023: ['Lipman et al., 2023', 'Flow Matching for Generative Modeling', 'ICLR 2023', 'https://arxiv.org/abs/2210.02747'],
  liu2023: ['Liu, Gong & Liu, 2023', 'Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow', 'ICLR 2023', 'https://arxiv.org/abs/2209.03003'],
  albergo2023: ['Albergo & Vanden-Eijnden, 2023', 'Building Normalizing Flows with Stochastic Interpolants', 'ICLR 2023', 'https://arxiv.org/abs/2209.15571'],
  esser2024: ['Esser et al., 2024', 'Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)', 'ICML 2024', 'https://arxiv.org/abs/2403.03206'],
  ma2024: ['Ma et al., 2024', 'SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers', 'ECCV 2024', 'https://arxiv.org/abs/2401.08740'],
  gao2024: ['Gao et al., 2024', 'Diffusion Meets Flow Matching: Two Sides of the Same Coin', 'blog post, 2024', 'https://diffusionflow.github.io/'],
  li2025: ['Li & He, 2025', 'Back to Basics: Let Denoising Generative Models Denoise', 'arXiv 2025', 'https://arxiv.org/abs/2511.13720'],
  kong2021: ['Kong et al., 2021', 'DiffWave: A Versatile Diffusion Model for Audio Synthesis', 'ICLR 2021', 'https://arxiv.org/abs/2009.09761'],
  chen2021: ['Chen et al., 2021', 'WaveGrad: Estimating Gradients for Waveform Generation', 'ICLR 2021', 'https://arxiv.org/abs/2009.00713'],
  popov2021: ['Popov et al., 2021', 'Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech', 'ICML 2021', 'https://arxiv.org/abs/2105.06337'],
  lee2022: ['Lee et al., 2022', 'PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior', 'ICLR 2022', 'https://arxiv.org/abs/2106.06406'],
  liu2023audio: ['Liu et al., 2023', 'AudioLDM: Text-to-Audio Generation with Latent Diffusion Models', 'ICML 2023', 'https://arxiv.org/abs/2301.12503'],
  evans2024: ['Evans et al., 2024', 'Fast Timing-Conditioned Latent Audio Diffusion (Stable Audio)', 'ICML 2024', 'https://arxiv.org/abs/2402.04825'],
  le2023: ['Le et al., 2023', 'Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale', 'NeurIPS 2023', 'https://arxiv.org/abs/2306.15687'],
  mehta2024: ['Mehta et al., 2024', 'Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching', 'ICASSP 2024', 'https://arxiv.org/abs/2309.03199'],
  chen2024: ['Chen et al., 2024', 'F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching', 'arXiv 2024', 'https://arxiv.org/abs/2410.06885'],
};

export function cite(key) {
  const r = REFS[key];
  if (!r) return key;
  return r[3] ? `<a href="${r[3]}" target="_blank" rel="noopener">${r[0]}</a>` : r[0];
}

export function bibliography() {
  return Object.values(REFS)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(r => `<p><b>${r[0]}.</b> ${r[1]}. <i style="font-family:inherit;font-size:1em">${r[2]}</i>.${r[3] ? ` <a href="${r[3]}" target="_blank" rel="noopener">link</a>` : ''}</p>`)
    .join('');
}

// ------------------------------------------------------------------ the three recipes
export const P1_RECIPES = [
  { key: 'target', label: 'Target', opts: [
    { v: 'eps', html: 'noise <i>ε</i>' },
    { v: 'x0', html: 'image <i>x</i><sub>0</sub>' },
  ] },
  { key: 'sched', label: 'Schedule', opts: [
    { v: 'linear', html: 'linear' },
    { v: 'cosine', html: 'cosine' },
    { v: 'sigmoid', html: 'sigmoid' },
  ] },
  { key: 'sampler', label: 'Sampler', opts: [
    { v: 'ddpm1000', html: 'DDPM · 1000' },
    { v: 'ddim50', html: 'DDIM · 50' },
    { v: 'ddim10', html: 'DDIM · 10' },
    { v: 'ddim5', html: 'DDIM · 5' },
  ], clip: true },
];

// ------------------------------------------------------------------ code panels
export function p1Code(target) {
  const tl = {
    eps: 'target = eps                          # DDPM: predict the noise',
    x0: 'target = x0                           # predict the clean image',
    v: 'target = a[t]*eps - s[t]*x0           # v: a mix of both',
  }[target];
  return [
    'x0  = next_batch()                     # clean images, pixels in [-1, 1]',
    't   = randint(1, T)                    # a random noise level',
    'eps = randn_like(x0)                   # the noise we hide',
    'xt  = a[t]*x0 + s[t]*eps               # a[t] = sqrt(alpha_bar[t]), s[t] = sqrt(1 - alpha_bar[t])',
    `<mark>${tl}</mark>`,
    'loss = ((net(xt, t) - target)**2).mean()',
  ].join('\n');
}

export function p2Code(s) {
  if (s.path === 'curved') return [
    'x0  = next_batch()',
    't   = randint(1, T)',
    'eps = randn_like(x0)',
    '<mark>xt  = a[t]*x0 + s[t]*eps               # curved path (cosine schedule)</mark>',
    '<mark>target = a[t]*eps - s[t]*x0           # v = velocity along the curve</mark>',
    'loss = ((net(xt, t) - target)**2).mean()',
  ].join('\n');
  return [
    'x0  = next_batch()',
    '<mark>t   = rand()                           # continuous time in [0, 1]</mark>',
    'eps = randn_like(x0)                   # start: Gaussian noise',
    '<mark>xt  = (1 - t)*x0 + t*eps               # straight line from data to start</mark>',
    '<mark>target = eps - x0                     # velocity: d xt / dt</mark>',
    'loss = ((net(xt, t) - target)**2).mean()',
  ].join('\n');
}

// ------------------------------------------------------------------ Part 2 static text
export const COMPARE = [
  ['Target', 'the noise <i>ε</i> (or <i>x</i><sub>0</sub>)', 'the velocity <i>u</i> = <i>ε</i> − <i>x</i><sub>0</sub>'],
  ['Schedule', '<i>x<sub>t</sub></i> = <i>a<sub>t</sub> x</i><sub>0</sub> + <i>s<sub>t</sub> ε</i>, a curve set by the schedule', '<i>x<sub>t</sub></i> = (1 − <i>t</i>) <i>x</i><sub>0</sub> + <i>t ε</i>, a straight line'],
  ['Sampler', '1000 random steps (DDPM), or DDIM', 'Euler: <i>x</i> ← <i>x</i> − Δ<i>t</i> · <i>û</i>, a few steps'],
];

export function meetCards() {
  return [
    ['DDIM on a straight line is Euler',
      `Put <i>a<sub>t</sub></i> = 1 − <i>t</i>, <i>s<sub>t</sub></i> = <i>t</i> into the DDIM update <i>x<sub>t′</sub></i> = <i>a<sub>t′</sub> x̂</i><sub>0</sub> + <i>s<sub>t′</sub> ε̂</i> and it becomes <i>x<sub>t</sub></i> − (<i>t</i> − <i>t′</i>) <i>û</i>: one Euler step. The “complicated coefficients” were the schedule. (${cite('gao2024')})`],
    ['v-prediction is a velocity',
      `On the cosine path <i>x</i> = cos φ · <i>x</i><sub>0</sub> + sin φ · <i>ε</i>, the derivative d<i>x</i>/dφ is exactly <i>v</i> = cos φ · <i>ε</i> − sin φ · <i>x</i><sub>0</sub>. A v-prediction network is a flow model on a curved path, which is why it is the curved path in recipe 2. (${cite('salimans2022')})`],
    ['The sampler is its own choice',
      `Diffusion models can be sampled without randomness (DDIM, the probability-flow ODE of ${cite('song2021sde')}), and flow models can be sampled with it (${cite('ma2024')}). Try the “stochastic” sampler in recipe 3.`],
  ];
}

// Famous models described by the same three recipes. `go` = the closest setting in this demo.
export const FAMOUS = [
  { name: 'DDPM', ref: 'ho2020', predicts: 'ε', sched: 'linear β, T = 1000', start: '<i>N</i>(0, I)', sampler: 'random (ancestral), 1000 steps',
    go: { part: 1, target: 'eps', sched: 'linear', sampler: 'ddpm1000' } },
  { name: 'DDIM', ref: 'song2021ddim', predicts: 'ε (same network)', sched: 'linear β', start: '<i>N</i>(0, I)', sampler: 'deterministic, 10–100 steps',
    go: { part: 1, target: 'eps', sched: 'linear', sampler: 'ddim50' } },
  { name: 'Improved DDPM', ref: 'nichol2021', predicts: 'ε (+ variance)', sched: 'cosine', start: '<i>N</i>(0, I)', sampler: 'random, 50–4000 steps',
    go: { part: 1, target: 'eps', sched: 'cosine', sampler: 'ddpm1000' } },
  { name: 'DiffWave', audio: true, ref: 'kong2021', predicts: 'ε', sched: 'linear β, T = 50', start: '<i>N</i>(0, I)', sampler: 'random; 6-step fast schedule',
    go: { part: 1, target: 'eps', sched: 'linear', sampler: 'ddpm1000' } },
  { name: 'Grad-TTS', audio: true, ref: 'popov2021', predicts: 'score (≈ −ε / s)', sched: 'VP SDE, linear β(t)', start: '<i>N</i>(μ, I), μ from the text', sampler: 'reverse-time solver, steps set at inference',
    go: null },
  { name: 'Stable Diffusion 1', ref: 'rombach2022', predicts: 'ε', sched: 'scaled linear β', start: '<i>N</i>(0, I) in a latent space', sampler: 'DDIM / PLMS, ~50 steps',
    go: { part: 1, target: 'eps', sched: 'linear', sampler: 'ddim50' } },
  { name: 'Stable Diffusion 2 (768-v)', ref: null, predicts: 'v', sched: 'scaled linear β', start: '<i>N</i>(0, I) in a latent space', sampler: 'DDIM-family, ~50 steps',
    go: null },
  { name: 'Stable Audio', audio: true, ref: 'evans2024', predicts: 'v', sched: 'cosine, continuous time', start: '<i>N</i>(0, I) in a latent space', sampler: 'DPM-Solver++, 100 steps',
    go: null },
  { name: 'Stable Diffusion 3', ref: 'esser2024', predicts: 'velocity', sched: 'straight line (rectified flow)', start: '<i>N</i>(0, I) in a latent space', sampler: 'Euler, ~28 steps',
    go: { part: 2, path: 'straight', sampler: 'euler', steps: 10 } },
  { name: 'Matcha-TTS', audio: true, ref: 'mehta2024', predicts: 'velocity', sched: 'straight line (OT-CFM)', start: '<i>N</i>(0, I)', sampler: 'Euler, 2–10 steps',
    go: { part: 2, path: 'straight', sampler: 'euler', steps: 5 } },
  { name: 'F5-TTS', audio: true, ref: 'chen2024', predicts: 'velocity', sched: 'straight line', start: '<i>N</i>(0, I)', sampler: 'Euler, 16–32 steps (“sway” spacing)',
    go: { part: 2, path: 'straight', sampler: 'euler', steps: 50 } },
];

export const TREE = [
  ['2015', 'sohl2015', 'Diffusion probabilistic models: destroy data with noise, learn to reverse it'],
  ['2019', 'song2019', 'Score matching across noise levels (NCSN)'],
  ['2020', 'ho2020', '<b>DDPM</b>: predict ε, linear schedule, 1000 random steps', true],
  ['2020', 'song2021ddim', '<b>DDIM</b>: same network, deterministic sampler, far fewer steps', true],
  ['2020', 'song2021sde', 'Score SDE: every diffusion model has a deterministic twin (probability-flow ODE)'],
  ['2021', 'nichol2021', 'Cosine schedule (Improved DDPM)'],
  ['2021', 'kong2021', 'Audio arrives: DiffWave, WaveGrad, Grad-TTS'],
  ['2022', 'salimans2022', '<b>v-prediction</b>, introduced for few-step distillation', true],
  ['2022', 'karras2022', 'EDM: the design space, untangled into separate choices'],
  ['2022', 'lipman2023', '<b>Flow matching</b>, rectified flow, stochastic interpolants (all at ICLR 2023)', true],
  ['2023', 'lin2024', 'Zero terminal SNR: fix the schedule, switch to v'],
  ['2023', 'le2023', 'Flow matching for speech: Voicebox, then Matcha-TTS and F5-TTS'],
  ['2024', 'esser2024', '<b>Stable Diffusion 3</b>: rectified flow at scale', true],
];
