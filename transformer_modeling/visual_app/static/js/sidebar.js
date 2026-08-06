/**
 * sidebar.js —— 左侧运行配置面板（硬件 / 请求 / 执行 / 并行 四个 Tab）
 *
 * 关键约定：
 *   界面单位是用户友好的（TOPS、GB、TB/s、GB/s、μs），config 里一律是原始单位
 *   （ops/s、bytes、bytes/s、seconds）。换算系数写在字段描述表的 factor 上。
 *   AppState.sidebarConfig 保存的是「已换算为原始单位」的 config 片段，
 *   且以 resolve-preset 返回的原始结构为底座，未在界面暴露的字段原样保留。
 */

import {
    AppState, Bus, EVENTS, UNITS, deepClone, getPath, setPath,
} from './api.js';
import {
    buildNumberRow, buildRangeRow, buildSelectRow, buildTextRow, buildToggleRow,
    hintEl, sectionEl,
} from './operator-panel.js';

const TABS = ['hardware', 'serving', 'execution', 'parallelism'];

/* ---------------------------------------------------------------- 字段描述表 */

/**
 * 字段描述：
 *   path    config 内的绝对路径（含 section 前缀）
 *   altPath 兼容路径（如 prompt_length 直接是数字而非 {distribution, value}）
 *   type    text | int | float | select | bool | range
 *   factor  界面值 → config 原始值 的乘数
 */
const FIELD_GROUPS = {
    hardware: [
        {
            title: '芯片与算力',
            fields: [
                { path: 'hardware.name', label: '芯片名称', type: 'text', fallback: 'custom-accelerator' },
                {
                    path: 'hardware.device_count', label: '设备数量', type: 'int', min: 1, fallback: 1,
                    hint: '必须等于 TP × EP × PP，否则后端会报错',
                },
                {
                    path: 'hardware.compute.throughput.fp16_dense_ops_per_second',
                    label: 'FP16 稠密算力', type: 'float', unit: 'TOPS',
                    factor: UNITS.TOPS, min: 0, step: 1, fallback: 100 * UNITS.TOPS,
                },
                {
                    path: 'hardware.compute.throughput.bf16_dense_ops_per_second',
                    label: 'BF16 dense throughput', type: 'float', unit: 'TOPS',
                    factor: UNITS.TOPS, min: 0, step: 1, fallback: 100 * UNITS.TOPS,
                },
                {
                    path: 'hardware.compute.throughput.fp8_dense_ops_per_second',
                    label: 'FP8 dense throughput', type: 'float', unit: 'TOPS',
                    factor: UNITS.TOPS, min: 0, step: 1, fallback: 200 * UNITS.TOPS,
                },
                {
                    path: 'hardware.compute.throughput.mxfp4_mxfp8_ops_per_second',
                    label: 'MXFP4 / MXFP8 throughput', type: 'float', unit: 'TOPS',
                    factor: UNITS.TOPS, min: 0, step: 1, fallback: 400 * UNITS.TOPS,
                    hint: 'Used by MXFP4 experts and MXFP8 work items; replace the placeholder with a measured or datasheet value.',
                },
            ],
        },
        {
            title: '显存 (HBM)',
            fields: [
                {
                    path: 'hardware.device_memory.capacity_bytes',
                    label: 'HBM 容量', type: 'float', unit: 'GB',
                    factor: UNITS.GB, min: 0, step: 1, fallback: 80 * UNITS.GB,
                    hint: '1 GB = 1024³ bytes',
                },
                {
                    path: 'hardware.device_memory.peak_bandwidth_bytes_per_second',
                    label: 'HBM 峰值带宽', type: 'float', unit: 'TB/s',
                    factor: UNITS.TBPS, min: 0, step: 0.1, fallback: 2 * UNITS.TBPS,
                },
                {
                    path: 'hardware.device_memory.reserved_capacity_bytes',
                    label: '运行时预留', type: 'float', unit: 'GB',
                    factor: UNITS.GB, min: 0, step: 0.5, fallback: UNITS.GB,
                    hint: '框架 / 上下文占用，不参与权重与 KV 分配',
                },
            ],
        },
        {
            title: '运行时与互联',
            fields: [
                {
                    path: 'hardware.runtime.kernel_launch_latency_seconds',
                    label: 'Kernel 启动延迟', type: 'float', unit: 'μs',
                    factor: UNITS.US, min: 0, step: 0.5, fallback: 5 * UNITS.US,
                },
                {
                    path: 'hardware.interconnect.topology', label: '互联拓扑', type: 'select',
                    options: ['ring', 'bus', 'crossbar', 'mesh'], fallback: 'ring',
                },
                {
                    path: 'hardware.interconnect.effective_channel_bandwidth_bytes_per_second',
                    label: '互联有效带宽', type: 'float', unit: 'GB/s',
                    factor: UNITS.GBPS, min: 0, step: 1, fallback: 100 * UNITS.GBPS,
                },
                {
                    path: 'hardware.interconnect.collective_step_latency_seconds',
                    label: '集合通信单步延迟', type: 'float', unit: 'μs',
                    factor: UNITS.US, min: 0, step: 0.5, fallback: 2 * UNITS.US,
                },
            ],
        },
    ],

    serving: [
        {
            title: '请求形状',
            fields: [
                { path: 'serving.batch_size', label: 'Batch Size', type: 'int', min: 1, fallback: 1 },
                {
                    path: 'serving.prompt_length.value', altPath: 'serving.prompt_length',
                    label: 'Prompt 长度', type: 'int', min: 1, fallback: 2048, unit: 'tokens',
                },
                {
                    path: 'serving.output_length.value', altPath: 'serving.output_length',
                    label: '输出长度', type: 'int', min: 1, fallback: 128, unit: 'tokens',
                },
                {
                    path: 'serving.max_sequence_length', label: '最大序列长度', type: 'int',
                    min: 1, fallback: 4096, unit: 'tokens',
                    hint: 'KV 缓存预算按该长度上限估算',
                },
            ],
        },
    ],

    execution: [
        {
            title: '算子融合',
            fields: [
                { path: 'execution.fusion.flash_attention', label: 'Flash Attention', type: 'bool', fallback: true },
                { path: 'execution.fusion.rope_kv_write', label: 'RoPE + KV 写入融合', type: 'bool', fallback: true },
                { path: 'execution.fusion.gated_mlp', label: 'Gated MLP 融合', type: 'bool', fallback: true },
                { path: 'execution.fusion.rmsnorm_linear', label: 'RMSNorm + Linear 融合', type: 'bool', fallback: true },
            ],
        },
        {
            title: '计算与通信重叠',
            fields: [
                {
                    path: 'execution.overlap.interpolation_rho', label: '重叠系数 ρ', type: 'range',
                    min: 0, max: 1, step: 0.05, fallback: 0.1,
                    hint: '0 = 完全串行，1 = 完全重叠',
                },
                { path: 'execution.overlap.tp_interpolation_rho', label: 'TP 重叠系数', type: 'range', min: 0, max: 1, step: 0.05, fallback: 1 },
                { path: 'execution.overlap.prefill_tp_interpolation_rho', label: 'Prefill TP 重叠', type: 'range', min: 0, max: 1, step: 0.05, fallback: 0.5 },
                { path: 'execution.overlap.decode_tp_interpolation_rho', label: 'Decode TP 重叠', type: 'range', min: 0, max: 1, step: 0.05, fallback: 1 },
                {
                    path: 'execution.overlap.ep_interpolation_rho', label: 'EP 重叠系数', type: 'range',
                    min: 0, max: 1, step: 0.05, fallback: 1,
                    hint: 'MoE 计算与 All-to-All 的重叠程度',
                },
            ],
        },
        {
            title: '效率系数',
            fields: [
                { path: 'execution.efficiencies.prefill_gemm', label: 'Prefill GEMM', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.65 },
                { path: 'execution.efficiencies.decode_gemm', label: 'Decode GEMM', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.2 },
                { path: 'execution.efficiencies.prefill_attention', label: 'Prefill Attention', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.5 },
                { path: 'execution.efficiencies.decode_attention', label: 'Decode Attention', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.15 },
                { path: 'execution.efficiencies.vector', label: '向量 / 逐元素', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.15 },
                { path: 'execution.efficiencies.hbm', label: 'HBM 带宽利用率', type: 'float', min: 0.01, max: 1, step: 0.01, fallback: 0.75 },
            ],
        },
        {
            title: 'KV 缓存',
            fields: [
                { path: 'execution.memory.kv_paged', label: '分页 KV 缓存', type: 'bool', fallback: true },
                { path: 'execution.memory.kv_page_tokens', label: '每页 token 数', type: 'int', min: 1, fallback: 16 },
            ],
        },
    ],

    parallelism: [
        {
            title: '并行切分',
            fields: [
                { path: 'parallelism.tensor_parallel', label: '张量并行 TP', type: 'int', min: 1, fallback: 1 },
                { path: 'parallelism.expert_parallel', label: '专家并行 EP', type: 'int', min: 1, fallback: 1, hint: '大于 1 时模型里必须含 MoE 算子' },
                { path: 'parallelism.pipeline_parallel', label: '流水并行 PP', type: 'int', min: 1, fallback: 1, hint: '不能超过模型层数' },
                {
                    path: 'parallelism.pipeline_microbatches', label: '流水微批数', type: 'int',
                    min: 1, fallback: 1, hint: 'batch_size 必须能被其整除',
                },
            ],
        },
        {
            title: 'KV 头切分策略',
            fields: [
                {
                    path: 'parallelism.kv_head_policy', label: 'KV Head 策略', type: 'select',
                    options: ['shard_or_group_replicate'],
                    fallback: 'shard_or_group_replicate',
                    hint: 'KV 头数够分时切分，不够时按组复制；当前引擎仅支持该策略',
                },
            ],
        },
    ],
};

const TAB_HINTS = {
    hardware: '硬件规格直接决定算力与带宽上限；单位已换算为工程常用口径。',
    serving: '请求形状决定 prefill / decode 的工作量与 KV 缓存占用。',
    execution: '融合、重叠与效率系数刻画真实执行相对理论峰值的折扣。',
    parallelism: '并行切分影响单卡权重、KV 分片与通信开销；设备数量需等于 TP × EP × PP。',
};

/* ---------------------------------------------------------------- 模块状态 */

let bodyEl = null;
let tabsEl = null;
let activeTab = 'hardware';

/** 初始化左侧栏 */
export function initSidebar(container, tabsContainer) {
    bodyEl = container;
    tabsEl = tabsContainer;

    tabsEl?.addEventListener('click', (event) => {
        const tab = event.target.closest('.tab');
        if (!tab) return;
        switchTab(tab.dataset.tab);
    });
    render();
}

/** 切换 Tab */
export function switchTab(tab) {
    if (!TABS.includes(tab)) return;
    activeTab = tab;
    for (const button of tabsEl?.querySelectorAll('.tab') ?? []) {
        button.classList.toggle('is-active', button.dataset.tab === tab);
    }
    render();
}

/**
 * 从完整 config 载入配置：以原始结构为底座，仅界面字段可被覆盖。
 */
export function loadFromConfig(config) {
    AppState.sidebarConfig = {
        hardware: deepClone(config?.hardware ?? {}),
        serving: deepClone(config?.serving ?? {}),
        execution: deepClone(config?.execution ?? {}),
        parallelism: deepClone(config?.parallelism ?? {}),
    };
    render();
}

/** 取出可直接提交后端的配置片段 */
export function getSidebarConfig() {
    return deepClone(AppState.sidebarConfig ?? {});
}

/* ---------------------------------------------------------------- 渲染 */

function render() {
    if (!bodyEl) return;
    bodyEl.innerHTML = '';

    if (!AppState.sidebarConfig || !Object.keys(AppState.sidebarConfig).length) {
        bodyEl.appendChild(hintEl('等待模型预设加载…'));
        return;
    }

    bodyEl.appendChild(hintEl(TAB_HINTS[activeTab] ?? ''));

    const groups = FIELD_GROUPS[activeTab] ?? [];
    groups.forEach((group, index) => {
        bodyEl.appendChild(buildAccordion(group, index === 0));
    });
}

/** 折叠面板（accordion）：默认展开第一个 */
function buildAccordion(group, open) {
    const details = document.createElement('details');
    details.className = 'acc';
    details.open = open !== false;

    const summary = document.createElement('summary');
    summary.className = 'acc-head';
    const title = document.createElement('span');
    title.textContent = group.title;
    const chevron = document.createElement('i');
    chevron.className = 'acc-chevron';
    summary.append(title, chevron);
    details.appendChild(summary);

    const box = document.createElement('div');
    box.className = 'acc-body';
    for (const field of group.fields) {
        box.appendChild(buildField(field));
    }
    details.appendChild(box);
    return details;
}

/** 按字段描述创建对应控件，change 时写回 AppState.sidebarConfig */
function buildField(field) {
    const factor = field.factor ?? 1;
    const raw = readRaw(field);
    const common = {
        label: field.label,
        hint: field.hint,
        code: leafName(field.path),
    };

    if (field.type === 'bool') {
        return buildToggleRow({
            ...common,
            value: raw === true,
            onChange: (value) => writeRaw(field, value),
        });
    }
    if (field.type === 'select') {
        const options = [...(field.options ?? [])];
        const current = raw ?? field.fallback;
        if (current != null && !options.includes(current)) options.unshift(current);
        return buildSelectRow({
            ...common,
            value: current,
            options,
            onChange: (value) => writeRaw(field, value),
        });
    }
    if (field.type === 'text') {
        return buildTextRow({
            ...common,
            value: raw ?? field.fallback ?? '',
            onChange: (value) => writeRaw(field, value),
        });
    }
    if (field.type === 'range') {
        return buildRangeRow({
            ...common,
            value: toDisplay(raw ?? field.fallback, factor),
            min: field.min ?? 0,
            max: field.max ?? 1,
            step: field.step ?? 0.05,
            onChange: (value) => writeRaw(field, value * factor),
        });
    }

    const isInt = field.type === 'int';
    return buildNumberRow({
        ...common,
        value: toDisplay(raw ?? field.fallback, factor),
        min: field.min,
        max: field.max,
        step: field.step ?? (isInt ? 1 : 0.01),
        unit: field.unit,
        onChange: (value) => {
            let next = value;
            if (field.min != null) next = Math.max(field.min, next);
            if (field.max != null) next = Math.min(field.max, next);
            if (isInt) next = Math.round(next);
            writeRaw(field, isInt ? next : next * factor);
        },
    });
}

/** 界面显示值 = 原始值 / factor，并做适度精度收敛 */
function toDisplay(rawValue, factor) {
    if (rawValue == null) return '';
    const num = Number(rawValue);
    if (!Number.isFinite(num)) return rawValue;
    if (factor === 1) return num;
    const scaled = num / factor;
    return Math.abs(scaled) >= 100 ? Math.round(scaled * 100) / 100 : Math.round(scaled * 10000) / 10000;
}

/** 读取原始值（兼容 altPath） */
function readRaw(field) {
    const config = AppState.sidebarConfig ?? {};
    const primary = getPath(config, field.path);
    if (primary !== undefined) return primary;
    if (field.altPath) {
        const alt = getPath(config, field.altPath);
        if (alt !== undefined && typeof alt !== 'object') return alt;
    }
    return undefined;
}

/** 写回原始值（若 config 里该项是标量而非对象，写到 altPath 上） */
function writeRaw(field, rawValue) {
    const config = AppState.sidebarConfig ?? (AppState.sidebarConfig = {});
    let targetPath = field.path;
    if (field.altPath) {
        const alt = getPath(config, field.altPath);
        if (alt !== undefined && (typeof alt !== 'object' || alt === null)) targetPath = field.altPath;
    }
    setPath(config, targetPath, rawValue);
    Bus.emit(EVENTS.CONFIG_CHANGED, { path: targetPath, value: rawValue });
}

/** 取路径末段作为字段代码展示 */
function leafName(path) {
    const parts = path.split('.');
    const last = parts[parts.length - 1];
    return last === 'value' ? parts.slice(-2).join('.') : last;
}
