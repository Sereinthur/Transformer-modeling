/**
 * results.js —— 底部估算结果面板
 *
 * 关键字段路径（后端 /api/estimate 实际结构）：
 *   capacity.capacity_feasible / available_bytes_per_rank
 *   capacity.per_stage[0].{weights_bytes, persistent_state_bytes, activations_peak_bytes, peak_total_bytes}
 *   performance.first_token.ttft_seconds
 *   performance.prefill.{latency_seconds, stages[].operators}
 *   performance.decode.{first_step, last_step, steps, total_latency_seconds, device_inter_token_interval}
 *   performance.throughput.{batch_average_output_tokens_per_second, steady_state_decode_output_tokens_per_second}
 *
 * 注意：decode 没有 latency_seconds，单步时延要从 first_step / last_step 里取；
 * 算子明细挂在各 PP Stage 下的 operators 数组，需要先展平。
 */

import { AppState, getPath, UNITS } from './api.js';

let bodyEl = null;
let summaryEl = null;

/** 初始化结果面板 */
export function initResults(container, summaryContainer) {
    bodyEl = container;
    summaryEl = summaryContainer ?? null;
}

/**
 * 渲染估算结果。
 * @param {object} results /api/estimate 返回体
 */
export function renderResults(results) {
    if (!bodyEl) return;
    AppState.results = results ?? AppState.results;
    const data = AppState.results;
    // 暴露到全局，方便在控制台查看被截断的完整返回体
    window.__results = data;
    bodyEl.innerHTML = '';
    if (!data) return;

    const capacity = data.capacity ?? {};
    const stages = capacity.per_stage ?? [];
    const criticalStageIndex = num(capacity.critical_stage_index);
    const stage = stages.find((item) => item.stage_index === criticalStageIndex)
        ?? stages[0] ?? {};
    const performance = data.performance ?? {};

    const weights = num(stage.weights_bytes);
    const persistent = num(stage.persistent_state_bytes);
    const activations = num(stage.activations_peak_bytes);
    const total = num(stage.peak_total_bytes) ?? sum([weights, persistent, activations]);
    const available = num(capacity.available_bytes_per_rank);
    const shortfall = num(capacity.capacity_shortfall_bytes) ?? 0;
    const feasible = capacity.capacity_feasible ?? data.validity?.capacity_feasible ?? (shortfall <= 0);

    const ttft = num(getPath(data, 'performance.first_token.ttft_seconds'))
        ?? num(getPath(data, 'performance.prefill.latency_seconds'));
    // 稳态单步解码时延：优先取设备层面的 token 间隔，其次末步 / 首步。
    const tpot = num(getPath(data, 'performance.decode.device_inter_token_interval'))
        ?? num(getPath(data, 'performance.decode.last_step.latency_seconds'))
        ?? num(getPath(data, 'performance.decode.first_step.latency_seconds'));
    const throughput = num(getPath(data, 'performance.throughput.batch_average_output_tokens_per_second'));
    const steady = num(getPath(data, 'performance.throughput.steady_state_decode_output_tokens_per_second'));
    const completion = num(getPath(data, 'performance.request.completion_latency_seconds'));
    const theoretical = data.validity?.performance_is_theoretical === true;

    bodyEl.appendChild(buildBanner({ feasible, theoretical, shortfall, model: data.model }));

    const grid = document.createElement('div');
    grid.className = 'results-grid';

    /* --- 容量总览 --- */
    const stageLabel = criticalStageIndex == null ? '单卡（关键 rank）显存占用'
        : `单卡（关键 rank：PP Stage ${criticalStageIndex}）显存占用`;
    const capCard = card('容量总览', stageLabel);
    capCard.appendChild(metricRow([
        metric('权重', bytesToGB(weights), 'GB', 'cap-weights'),
        metric('KV / 持久状态', bytesToGB(persistent), 'GB', 'cap-kv'),
        metric('激活峰值', bytesToGB(activations), 'GB', 'cap-act'),
        metric('峰值总计', bytesToGB(total), 'GB', 'cap-total'),
    ]));
    capCard.appendChild(buildCapacityBar({ weights, persistent, activations, total, available }));
    capCard.appendChild(buildCapacityFooter({ total, available, shortfall, headroom: num(capacity.headroom_bytes), feasible }));
    grid.appendChild(capCard);

    /* --- 性能指标 --- */
    const perfCard = card('性能指标', '端到端时延与吞吐');
    perfCard.appendChild(metricRow([
        metric('TTFT 首 token', secondsToMs(ttft), 'ms', 'perf-ttft'),
        metric('TPOT 单步解码', secondsToMs(tpot), 'ms', 'perf-tpot'),
        metric('输出吞吐', fixed(throughput, 1), 'tok/s', 'perf-tps'),
    ]));
    const extras = [];
    const prefill = num(getPath(data, 'performance.prefill.latency_seconds'));
    const decodeTotal = num(getPath(data, 'performance.decode.total_latency_seconds'));
    const decodeSteps = num(getPath(data, 'performance.decode.steps'));
    if (prefill != null) extras.push(['Prefill 时延', `${secondsToMs(prefill)} ms`]);
    if (decodeTotal != null) {
        extras.push(['Decode 总时延', decodeSteps != null
            ? `${fixed(decodeTotal, 3)} s（${decodeSteps} 步）`
            : `${fixed(decodeTotal, 3)} s`]);
    }
    if (steady != null) extras.push(['稳态解码吞吐', `${fixed(steady, 1)} tok/s`]);
    if (completion != null) extras.push(['请求完成时延', `${fixed(completion, 3)} s`]);
    if (extras.length) perfCard.appendChild(kvList(extras));
    grid.appendChild(perfCard);

    /* --- 模型概况 --- */
    if (data.model) {
        const modelCard = card('模型概况', data.model.id ?? '');
        const modelRows = [
            ['名称', data.model.name ?? '—'],
            ['层数', data.model.layers ?? '—'],
            ['Hidden Size', data.model.hidden_size ?? '—'],
            ['已建模参数量', data.model.parameters != null ? compact(data.model.parameters) : '—'],
            ['模型默认精度', `权重 ${data.model.weight_dtype ?? '—'} · 激活 ${data.model.activation_dtype ?? '—'} · KV ${data.model.kv_cache_dtype ?? '—'} · 状态 ${data.model.state_dtype ?? '—'}`],
            ['结构精度', data.model.structure_accuracy ?? '—'],
            ['性能公式置信度', data.model.performance_formula_confidence ?? '—'],
        ];
        const weightBuckets = Object.entries(stage.weights_by_dtype ?? {})
            .filter(([, value]) => Number(value) > 0)
            .map(([dtype, value]) => `${dtype} ${bytesToGB(value)} GB`);
        if (weightBuckets.length) modelRows.push(['本卡权重分桶', weightBuckets.join(' · ')]);
        const specialOperators = Object.entries(data.model.special_operator_estimation ?? {})
            .map(([type, value]) => `${type} ×${value.occurrences}（${value.formula_confidence}）`);
        if (specialOperators.length) modelRows.push(['特殊状态算子', specialOperators.join(' · ')]);
        const reference = data.model.metadata?.published_parameter_reference;
        if (reference) modelRows.push(['官方参数参考', compact(reference)]);
        const fallbackFormats = data.performance?.compute_throughput_fallback_formats ?? [];
        if (fallbackFormats.length) {
            modelRows.push(['硬件吞吐回退', fallbackFormats.join(', ')]);
        }
        modelCard.appendChild(kvList(modelRows));
        grid.appendChild(modelCard);

    }

    bodyEl.appendChild(grid);

    /* --- 警告 --- */
    const warnings = distinctWarnings(data.warnings);
    // Capacity feasibility is already presented once in the verdict banner and
    // capacity card. Keep the API/CLI warning, but do not repeat it in the UI.
    const additionalWarnings = feasible ? warnings : warnings.filter(
        (warning) => !isCapacityWarning(warning),
    );
    if (additionalWarnings.length) bodyEl.appendChild(buildWarnings(additionalWarnings));

    /* --- 算子级分解（逐 PP Stage 展平）--- */
    const prefillPhase = getPath(data, 'performance.prefill');
    const decodePhase = getPath(data, 'performance.decode.first_step');
    const prefillOps = flattenOperators(prefillPhase);
    const decodeOps = flattenOperators(decodePhase);
    if (prefillOps.length) {
        bodyEl.appendChild(buildOperatorTable('Prefill 算子分解', prefillOps, num(prefillPhase?.latency_seconds)));
    }
    if (decodeOps.length) {
        bodyEl.appendChild(buildOperatorTable('Decode 算子分解（单步）', decodeOps, num(decodePhase?.latency_seconds)));
    }

    /* --- 原始 JSON --- */
    bodyEl.appendChild(buildRawJson(data));

    renderSummary({ feasible, total, available, ttft, tpot, throughput });
}

/** 顶部摘要（结果面板折叠时也能看到关键结论） */
function renderSummary({ feasible, total, available, ttft, tpot, throughput }) {
    if (!summaryEl) return;
    summaryEl.innerHTML = '';
    const items = [
        ['TTFT', `${secondsToMs(ttft)} ms`],
        ['TPOT', `${secondsToMs(tpot)} ms`],
        ['吞吐', `${fixed(throughput, 1)} tok/s`],
        ['显存', `${bytesToGB(total)} / ${bytesToGB(available)} GB`],
    ];
    for (const [key, value] of items) {
        const chip = document.createElement('span');
        chip.className = 'sum-chip';
        const k = document.createElement('em');
        k.textContent = key;
        const v = document.createElement('b');
        v.textContent = value;
        chip.append(k, v);
        summaryEl.appendChild(chip);
    }
    const flag = document.createElement('span');
    flag.className = `sum-flag ${feasible ? 'is-ok' : 'is-bad'}`;
    flag.textContent = feasible ? '容量可行' : '容量超限';
    summaryEl.appendChild(flag);
}

/* ---------------------------------------------------------------- 结构块 */

function buildBanner({ feasible, theoretical, shortfall, model }) {
    const banner = document.createElement('div');
    banner.className = `verdict ${feasible ? 'is-ok' : 'is-bad'}`;

    const glyph = document.createElement('span');
    glyph.className = 'verdict-glyph';
    glyph.textContent = feasible ? '✓' : '!';

    const text = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = feasible ? '配置可行：显存容量满足需求' : '配置不可行：显存容量超出上限';
    const detail = document.createElement('p');
    if (feasible) {
        detail.textContent = model?.name
            ? `${model.name} 在当前硬件与并行策略下可部署。`
            : '当前硬件与并行策略下可部署。';
    } else {
        detail.textContent = `仍缺少 ${bytesToGB(shortfall)} GB 显存，请调整并行度、batch 或序列长度${theoretical ? '；性能数值为理论参考值' : ''}。`;
    }
    text.append(title, detail);
    banner.append(glyph, text);
    return banner;
}

/** 容量条形图：权重 / KV / 激活堆叠，超出可用容量时整条转红 */
function buildCapacityBar({ weights, persistent, activations, total, available }) {
    const wrap = document.createElement('div');
    wrap.className = 'cap-bar-wrap';

    const parts = [
        ['cap-weights', '权重', weights ?? 0],
        ['cap-kv', 'KV / 持久状态', persistent ?? 0],
        ['cap-act', '激活', activations ?? 0],
    ];
    const used = total ?? sum(parts.map((p) => p[2]));
    const scale = Math.max(used, available ?? 0) || 1;

    const bar = document.createElement('div');
    bar.className = `cap-bar${available != null && used > available ? ' is-over' : ''}`;
    for (const [cls, label, value] of parts) {
        if (!value) continue;
        const seg = document.createElement('div');
        seg.className = `cap-seg ${cls}`;
        seg.style.width = `${(value / scale) * 100}%`;
        seg.title = `${label}：${bytesToGB(value)} GB`;
        bar.appendChild(seg);
    }
    // HBM 可用容量刻度线
    if (available != null) {
        const mark = document.createElement('div');
        mark.className = 'cap-mark';
        mark.style.left = `${Math.min(100, (available / scale) * 100)}%`;
        mark.dataset.label = `HBM ${bytesToGB(available)} GB`;
        bar.appendChild(mark);
    }
    wrap.appendChild(bar);

    const legend = document.createElement('div');
    legend.className = 'cap-legend';
    for (const [cls, label, value] of parts) {
        const item = document.createElement('span');
        item.className = `cap-legend-item ${cls}`;
        item.textContent = `${label} ${bytesToGB(value)} GB`;
        legend.appendChild(item);
    }
    wrap.appendChild(legend);
    return wrap;
}

function buildCapacityFooter({ total, available, shortfall, headroom, feasible }) {
    const rows = [];
    if (available != null) rows.push(['HBM 可用', `${bytesToGB(available)} GB`]);
    if (total != null && available) {
        rows.push(['占用率', `${fixed((total / available) * 100, 1)} %`]);
    }
    if (feasible) {
        if (headroom != null) rows.push(['剩余余量', `${bytesToGB(headroom)} GB`]);
    } else if (shortfall) {
        rows.push(['容量缺口', `${bytesToGB(shortfall)} GB`]);
    }
    return kvList(rows);
}

function distinctWarnings(warnings) {
    const seen = new Set();
    return (Array.isArray(warnings) ? warnings : []).filter((warning) => {
        const text = typeof warning === 'string' ? warning.trim() : safeJson(warning);
        if (!text || seen.has(text)) return false;
        seen.add(text);
        return true;
    });
}

function isCapacityWarning(warning) {
    return typeof warning === 'string'
        && /容量不足|容量超限|容量不可行|理论值/.test(warning);
}

function buildWarnings(warnings) {
    const box = document.createElement('div');
    box.className = 'warn-box';
    const title = document.createElement('h4');
    title.className = 'section-title';
    title.textContent = `其他告警（${warnings.length}）`;
    box.appendChild(title);
    const list = document.createElement('ul');
    list.className = 'warn-list';
    for (const item of warnings) {
        const li = document.createElement('li');
        li.textContent = typeof item === 'string' ? item : safeJson(item);
        list.appendChild(li);
    }
    box.appendChild(list);
    return box;
}

/**
 * 把一个阶段节点（prefill 或 decode.first_step）下各 PP Stage 的算子明细展平。
 * 多 Stage 时额外标上 stage 序号，方便定位流水切分。
 */
function flattenOperators(phase) {
    const stages = phase?.stages;
    if (!Array.isArray(stages)) return [];
    const rows = [];
    stages.forEach((stage, index) => {
        for (const operator of stage?.operators ?? []) {
            rows.push(stages.length > 1 ? { ...operator, stage_index: index } : operator);
        }
    });
    return rows;
}

/** 算子行的估计耗时（含通信） */
function operatorSeconds(row) {
    return num(getPath(row, 'time_seconds.estimated'))
        ?? num(getPath(row, 'time_seconds.local'))
        ?? 0;
}

const CONFIDENCE_TEXT = { high: '高', medium: '中', low: '低' };

/**
 * 算子表列定义：直接对齐后端算子行结构
 * {type, 中文名称, occurrences, work:{...}, time_seconds:{...}, capacity:{...}, confidence}。
 * when 为真时才显示该列，避免单卡场景出现恒为 0 的空列。
 */
const OP_COLUMNS = [
    { label: '算子', get: (row) => row['中文名称'] ?? row.type ?? '—' },
    { label: '类型', get: (row) => row.type ?? '—', mono: true },
    {
        label: 'Stage', num: true,
        when: (rows) => rows.some((row) => row.stage_index != null),
        get: (row) => (row.stage_index == null ? '—' : String(row.stage_index)),
    },
    { label: '出现次数', num: true, get: (row) => (row.occurrences == null ? '—' : String(row.occurrences)) },
    { label: '时延 (ms)', num: true, get: (row) => secondsToMs(operatorSeconds(row)) },
    {
        label: '通信 (ms)', num: true,
        when: (rows) => rows.some((row) => num(getPath(row, 'time_seconds.communication'))),
        get: (row) => secondsToMs(getPath(row, 'time_seconds.communication')),
    },
    { label: 'OPs', num: true, get: (row) => compact(getPath(row, 'work.executed_ops')) },
    { label: '访存 (GB)', num: true, get: (row) => bytesToGB(getPath(row, 'work.hbm_payload_bytes')) },
    {
        label: '计算格式',
        when: (rows) => rows.some((row) => (row.suboperators ?? []).some((item) => item.compute_dtype)),
        get: (row) => [...new Set((row.suboperators ?? []).map((item) => item.compute_dtype).filter(Boolean))].join(' / ') || '—',
    },
    {
        label: '吞吐来源',
        when: (rows) => rows.some((row) => (row.suboperators ?? []).some((item) => item.compute_throughput_source)),
        get: (row) => [...new Set((row.suboperators ?? []).map((item) => item.compute_throughput_source).filter(Boolean))].join(' / ') || '—',
    },
    { label: '本卡参数', num: true, get: (row) => compact(getPath(row, 'capacity.local_parameters')) },
    {
        label: '持久状态 (GB)', num: true,
        when: (rows) => rows.some((row) => num(getPath(row, 'capacity.persistent_state_bytes'))),
        get: (row) => bytesToGB(getPath(row, 'capacity.persistent_state_bytes')),
    },
    { label: '置信度', get: (row) => CONFIDENCE_TEXT[row.confidence] ?? row.confidence ?? '—' },
];

/** 算子级分解表：按耗时降序，末列给占比条 */
function buildOperatorTable(titleText, operators, totalSeconds) {
    const details = document.createElement('details');
    details.className = 'acc acc-table';

    const summary = document.createElement('summary');
    summary.className = 'acc-head';
    const title = document.createElement('span');
    title.textContent = `${titleText}（${operators.length} 项）`;
    const chevron = document.createElement('i');
    chevron.className = 'acc-chevron';
    summary.append(title, chevron);
    details.appendChild(summary);

    const rows = [...operators].sort((a, b) => operatorSeconds(b) - operatorSeconds(a));
    const columns = OP_COLUMNS.filter((column) => !column.when || column.when(rows));

    const scroll = document.createElement('div');
    scroll.className = 'table-scroll';
    const table = document.createElement('table');
    table.className = 'op-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const column of columns) {
        const th = document.createElement('th');
        th.textContent = column.label;
        headRow.appendChild(th);
    }
    if (totalSeconds) {
        const th = document.createElement('th');
        th.textContent = '占比';
        headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const row of rows) {
        const tr = document.createElement('tr');
        for (const column of columns) {
            const td = document.createElement('td');
            td.textContent = column.get(row);
            if (column.num || column.mono) td.classList.add('is-num');
            tr.appendChild(td);
        }
        if (totalSeconds) {
            const share = operatorSeconds(row) / totalSeconds;
            const td = document.createElement('td');
            td.className = 'share-cell';
            const track = document.createElement('span');
            track.className = 'share-bar';
            track.style.width = `${Math.min(100, share * 100)}%`;
            const label = document.createElement('em');
            label.textContent = `${fixed(share * 100, 1)}%`;
            td.append(track, label);
            tr.appendChild(td);
        }
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    scroll.appendChild(table);
    details.appendChild(scroll);
    return details;
}

/** 原始 JSON 展示上限：真实响应含每个算子的 suboperators，全量可达数 MB */
const RAW_JSON_LIMIT = 200000;

function buildRawJson(data) {
    const details = document.createElement('details');
    details.className = 'acc acc-raw';
    const summary = document.createElement('summary');
    summary.className = 'acc-head';
    const title = document.createElement('span');
    title.textContent = '原始返回 JSON';
    const chevron = document.createElement('i');
    chevron.className = 'acc-chevron';
    summary.append(title, chevron);
    details.appendChild(summary);

    const text = safeJson(data, 2);
    if (text.length > RAW_JSON_LIMIT) {
        title.textContent = `原始返回 JSON（共 ${Math.round(text.length / 1024)} KB，已截断预览）`;
        const tip = document.createElement('p');
        tip.className = 'hint raw-tip';
        tip.textContent = '内容过长，此处仅展示前 200 KB；完整数据可在浏览器控制台用 window.__results 查看。';
        details.appendChild(tip);
    }

    const pre = document.createElement('pre');
    pre.className = 'raw-json';
    pre.textContent = text.length > RAW_JSON_LIMIT
        ? `${text.slice(0, RAW_JSON_LIMIT)}\n… 已截断 …`
        : text;
    details.appendChild(pre);
    return details;
}

/* ---------------------------------------------------------------- 基础组件 */

function card(titleText, subText) {
    const box = document.createElement('section');
    box.className = 'res-card';
    const head = document.createElement('header');
    const title = document.createElement('h4');
    title.textContent = titleText;
    head.appendChild(title);
    if (subText) {
        const sub = document.createElement('span');
        sub.className = 'hint';
        sub.textContent = subText;
        head.appendChild(sub);
    }
    box.appendChild(head);
    return box;
}

function metricRow(metrics) {
    const row = document.createElement('div');
    row.className = 'metric-row';
    metrics.forEach((item) => row.appendChild(item));
    return row;
}

function metric(label, value, unit, cls) {
    const box = document.createElement('div');
    box.className = `metric ${cls ?? ''}`;
    const v = document.createElement('strong');
    v.textContent = value;
    const u = document.createElement('span');
    u.className = 'metric-unit';
    u.textContent = unit;
    const l = document.createElement('span');
    l.className = 'metric-label';
    l.textContent = label;
    const line = document.createElement('div');
    line.className = 'metric-value';
    line.append(v, u);
    box.append(line, l);
    return box;
}

function kvList(rows) {
    const list = document.createElement('dl');
    list.className = 'kv-list';
    for (const [key, value] of rows) {
        const dt = document.createElement('dt');
        dt.textContent = key;
        const dd = document.createElement('dd');
        dd.textContent = String(value);
        list.append(dt, dd);
    }
    return list;
}

/* ---------------------------------------------------------------- 格式化 */

function num(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function sum(values) {
    const valid = values.filter((v) => Number.isFinite(v));
    return valid.length ? valid.reduce((a, b) => a + b, 0) : null;
}

function bytesToGB(bytes) {
    const value = num(bytes);
    if (value == null) return '—';
    const gb = value / UNITS.GB;
    return gb >= 100 ? gb.toFixed(1) : gb.toFixed(2);
}

function secondsToMs(seconds) {
    const value = num(seconds);
    if (value == null) return '—';
    const ms = value / UNITS.MS;
    if (ms >= 100) return ms.toFixed(1);
    if (ms >= 1) return ms.toFixed(2);
    return ms.toFixed(3);
}

function fixed(value, digits) {
    const parsed = num(value);
    return parsed == null ? '—' : parsed.toFixed(digits);
}

function compact(value) {
    const parsed = num(value);
    if (parsed == null) return '—';
    if (parsed >= 1e12) return `${(parsed / 1e12).toFixed(2)} T`;
    if (parsed >= 1e9) return `${(parsed / 1e9).toFixed(2)} B`;
    if (parsed >= 1e6) return `${(parsed / 1e6).toFixed(2)} M`;
    return String(parsed);
}

function safeJson(value, indent = 0) {
    try {
        return JSON.stringify(value, null, indent);
    } catch {
        return String(value);
    }
}
