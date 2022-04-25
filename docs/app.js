/* Utilidades de formatação */
const fmtBRL = (v) => v.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
const fmtPct = (v) => `${(v * 100).toFixed(0)}%`;

/* Estado do modelo carregado */
let MODEL = null;

async function loadModel() {
  const res = await fetch('model_params.json');
  if (!res.ok) throw new Error('Falha ao carregar model_params.json');
  MODEL = await res.json();
}

/* Escalonadores (replicam scikit-learn) */
function minmax(x, min, max) {
  const denom = (max - min) || 1e-8;
  return (x - min) / denom;
}
function inverseMinmax(xScaled, min, max) {
  return xScaled * (max - min) + min;
}
function robust(x, center, scale) {
  const s = scale || 1e-8;
  return (x - center) / s;
}

/* Predição da quantidade usando LR em espaço escalonado */
function predictQuantity(precoVenda, precoOriginal, desconto) {
  const mf = MODEL.scalers.minmax_features;
  const rd = MODEL.scalers.robust_desconto;
  const mt = MODEL.scalers.minmax_target;
  const [pvMin, poMin] = mf.data_min;
  const [pvMax, poMax] = mf.data_max;
  const dCenter = rd.center[0];
  const dScale = rd.scale[0] || 1e-8;

  const xScaled = [
    minmax(precoVenda, pvMin, pvMax),
    minmax(precoOriginal, poMin, poMax),
    robust(desconto, dCenter, dScale),
  ];

  const { coefficients: coefs, intercept } = MODEL.linear_regression;
  let yScaled = intercept;
  for (let i = 0; i < coefs.length; i++) yScaled += coefs[i] * xScaled[i];

  const y = inverseMinmax(yScaled, mt.data_min, mt.data_max);
  return Math.max(0, y); // clamp não-negativo
}

/* Atualiza UI para um desconto específico */
function updateForDiscount(d) {
  const precoOriginal = parseFloat(document.getElementById('precoOriginal').value) || 0;
  const precoVenda = precoOriginal * (1 - d);
  const q = predictQuantity(precoVenda, precoOriginal, d);
  const rev = precoVenda * q;
  document.getElementById('priceOutput').textContent = fmtBRL(precoVenda);
  document.getElementById('quantityOutput').textContent = q.toFixed(2);
  document.getElementById('revenueOutput').textContent = fmtBRL(rev);
}

/* Busca o melhor desconto via grade */
function optimize() {
  const precoOriginal = parseFloat(document.getElementById('precoOriginal').value) || 0;
  const comp = parseFloat(document.getElementById('competitorPrice').value);
  let min = 0.0, max = 0.5;
  if (!Number.isNaN(comp) && comp > 0 && precoOriginal > 0) {
    const required = Math.max(0, 1 - (comp / precoOriginal));
    max = Math.min(1.0, Math.max(max, required));
  }
  let best = { d: min, price: 0, q: 0, rev: -Infinity };
  const steps = 101;
  for (let i = 0; i < steps; i++) {
    const d = min + (i * (max - min) / (steps - 1));
    const pv = precoOriginal * (1 - d);
    const q = predictQuantity(pv, precoOriginal, d);
    const rev = pv * q;
    if (rev > best.rev) best = { d, price: pv, q, rev };
  }
  // Atualiza bloco ótimo e sincroniza slider
  document.getElementById('bestDiscount').textContent = fmtPct(best.d);
  document.getElementById('bestPrice').textContent = fmtBRL(best.price);
  document.getElementById('bestQuantity').textContent = best.q.toFixed(2);
  document.getElementById('bestRevenue').textContent = fmtBRL(best.rev);
  const slider = document.getElementById('discount');
  slider.value = best.d.toFixed(2);
  document.getElementById('discountLabel').textContent = fmtPct(best.d);
  updateForDiscount(best.d);
}

async function bootstrap() {
  await loadModel();
  const slider = document.getElementById('discount');
  const label = document.getElementById('discountLabel');
  const onChange = () => {
    const d = parseFloat(slider.value);
    label.textContent = fmtPct(d);
    updateForDiscount(d);
  };
  slider.addEventListener('input', onChange);
  document.getElementById('precoOriginal').addEventListener('input', () => onChange());
  document.getElementById('competitorPrice').addEventListener('input', () => onChange());
  document.getElementById('optimizeBtn').addEventListener('click', optimize);
  onChange(); // inicial
}

bootstrap().catch((e) => {
  console.error(e);
  alert('Falha ao inicializar o app. Verifique se model_params.json existe.');
});