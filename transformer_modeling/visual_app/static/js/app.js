/**
 * app.js —— 应用主入口：生命周期、工具栏交互、全局错误处理与 Toast
 */

import {
    ApiError, AppState, Bus, EVENTS,
    configToFlowchart, fetchModelPresets, fetchOperatorCatalog, fetchOperatorSchemas,
    flowchartToConfig, resolvePreset, runEstimate,
} from './api.js';
import {
    fitView, initFlowchart, renderFlowchart, resetNodeOffsets, setAllGroupsCollapsed, setModelLayerCount, zoomBy,
} from './flowchart.js';
import {
    LINEAR_LAYOUT_DEFAULTS, applyLinearLayoutOverrides, getLinearLayout, resetLinearLayout,
} from './layout.js';
import { initOperatorPanel, renderOperatorPanel } from './operator-panel.js';
import { getSidebarConfig, initSidebar, loadFromConfig } from './sidebar.js';
import { initResults, renderResults } from './results.js';

/** 常用 DOM 引用 */
const dom = {};
let resultsStale = false;

/* ------------------------------------------------ 布局参数持久化 */

/** 布局覆盖参数的 localStorage 键名 */
const LAYOUT_STORAGE_KEY = 'vm-linear-layout';
/** 布局弹层暴露的参数（其余默认键不在面板内编辑） */
const LAYOUT_PANEL_KEYS = ['cardGap', 'groupGap', 'cardW'];

/** 启动时从 localStorage 恢复布局覆盖；异常/非法数据静默回落默认值 */
function loadStoredLinearLayout() {
    try {
        const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
        const clean = {};
        for (const key of Object.keys(LINEAR_LAYOUT_DEFAULTS)) {
            const value = parsed[key];
            if (typeof value === 'number' && Number.isFinite(value)) clean[key] = value;
        }
        applyLinearLayoutOverrides(clean); // 内部仅接受已知键，getLinearLayout 时再铳制
    } catch {
        // 存储不可用或 JSON 损坏：忽略，使用默认布局
    }
}

/** 面板变更后把当前生效的三个面板参数（cardGap / groupGap / cardW）写回 localStorage */
function saveLinearLayout() {
    try {
        const layout = getLinearLayout();
        const stored = {};
        for (const key of LAYOUT_PANEL_KEYS) stored[key] = layout[key];
        localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(stored));
    } catch {
        // 存储不可用时不阻断交互
    }
}

/* ---------------------------------------------------------------- 启动 */

async function init() {
    cacheDom();
    loadStoredLinearLayout(); // 需在首次渲染前恢复布局覆盖
    initFlowchart(dom.canvas);
    initOperatorPanel(dom.inspectorBody, dom.inspectorTag);
    initSidebar(dom.sidebarBody, dom.sidebarTabs);
    initResults(dom.resultsBody, dom.resultsSummary);
    bindToolbar();
    bindBus();
    bindGlobalErrors();

    try {
        await withLoading('加载算子目录…', async () => {
            const [catalog, schemas] = await Promise.all([
                fetchOperatorCatalog(),
                fetchOperatorSchemas(),
            ]);
            AppState.operatorCatalog = catalog;
            AppState.operatorSchemas = schemas;
        });
    } catch (error) {
        reportError(error, '算子目录加载失败，算子替换与参数表单将不可用');
    }

    try {
        const presets = await withLoading('加载模型预设…', fetchModelPresets);
        fillPresetOptions(presets);
        if (presets.length) {
            await loadPreset(presets[0].id);
        } else {
            toast('info', '后端未返回任何模型预设');
        }
    } catch (error) {
        reportError(error, '模型预设加载失败');
    }
}

function cacheDom() {
    dom.presetSelect = document.getElementById('preset-select');
    dom.presetDesc = document.getElementById('preset-desc');
    dom.btnCompute = document.getElementById('btn-compute');
    dom.btnImport = document.getElementById('btn-import');
    dom.btnExport = document.getElementById('btn-export');
    dom.fileImport = document.getElementById('file-import');
    dom.workspace = document.getElementById('workspace');
    dom.sidebar = document.getElementById('sidebar');
    dom.sidebarBody = document.getElementById('sidebar-body');
    dom.sidebarTabs = document.getElementById('sidebar-tabs');
    dom.sidebarToggle = document.getElementById('btn-sidebar-toggle');
    dom.sidebarRail = document.getElementById('sidebar-rail');
    dom.canvas = document.getElementById('flow-canvas');
    dom.modelInfo = document.getElementById('model-info');
    dom.zoomLevel = document.getElementById('zoom-level');
    dom.inspectorBody = document.getElementById('inspector-body');
    dom.inspectorTag = document.getElementById('inspector-tag');
    dom.resultsPanel = document.getElementById('results-panel');
    dom.resultsBody = document.getElementById('results-body');
    dom.resultsSummary = document.getElementById('results-summary');
    dom.resultsToggle = document.getElementById('btn-results-toggle');
    dom.toastStack = document.getElementById('toast-stack');
    dom.loadingVeil = document.getElementById('loading-veil');
    dom.loadingText = document.getElementById('loading-text');
    dom.btnLayout = document.getElementById('btn-layout');
    dom.layoutPopover = document.getElementById('layout-popover');
    dom.btnLayoutReset = document.getElementById('btn-layout-reset');
    dom.btnResetOffsets = document.getElementById('btn-reset-offsets');
}

/* ---------------------------------------------------------------- 预设加载 */

/** 映射质量代码 → 中文说明（与后端 metadata.mapping_quality 对齐） */
const MAPPING_QUALITY_TEXT = {
    exact_for_supported_fields: '已支持字段精确对应',
    field_mapping: 'HF 字段直接映射',
    approximate: '近似映射',
    parameterized_draft: '参数化草案',
    official_config_aligned: '官方配置对齐',
    customized_from_official: '基于官方配置的自定义模型',
};

/** 预设摘要：系列 + 映射质量 + 未建模特性数量 */
function presetSummary(preset) {
    if (!preset) return '';
    const parts = [];
    if (preset.family) parts.push(preset.family);
    const quality = MAPPING_QUALITY_TEXT[preset.mapping_quality] ?? preset.mapping_quality;
    if (quality) parts.push(quality);
    const unsupported = preset.unsupported_features ?? [];
    if (unsupported.length) parts.push(`未建模 ${unsupported.length} 项`);
    return parts.join(' · ');
}

function fillPresetOptions(presets) {
    dom.presetSelect.innerHTML = '';
    for (const preset of presets) {
        const option = document.createElement('option');
        option.value = preset.id;
        option.textContent = preset.name ?? preset.id;
        option.title = presetSummary(preset);
        dom.presetSelect.appendChild(option);
    }
    if (!presets.length) {
        const option = document.createElement('option');
        option.textContent = '（无可用预设）';
        dom.presetSelect.appendChild(option);
    }
}

/** 预设 → config → 流程图 */
async function loadPreset(presetId) {
    await withLoading('解析模型预设…', async () => {
        const config = await resolvePreset(presetId);
        applyConfig(config);
        AppState.currentPresetId = presetId;
        if (dom.presetSelect.value !== presetId) dom.presetSelect.value = presetId;
        const preset = AppState.presets.find((item) => item.id === presetId);
        const summary = presetSummary(preset);
        dom.presetDesc.textContent = summary;
        dom.presetDesc.title = summary;
        const flowchart = await configToFlowchart(config);
        renderFlowchart(flowchart);
        renderModelInfo(flowchart.model_info);
        renderOperatorPanel(null);
        AppState.selectedNode = null;
        clearResults();
    });
    // 预设自带的未建模特性提示，直接透传给用户，避免误读估算精度。
    const warnings = AppState.presetWarnings ?? [];
    if (warnings.length) {
        toast('info', `该预设有未建模特性：${warnings.join('、')}`);
    }
}

/** 把完整 config 落到状态与左侧栏 */
function applyConfig(config) {
    AppState.baseConfig = config;
    loadFromConfig(config);
}

/** 顶部模型概况条 */
function renderModelInfo(info) {
    dom.modelInfo.innerHTML = '';
    if (!info) {
        const span = document.createElement('span');
        span.className = 'mi-name';
        span.textContent = '—';
        dom.modelInfo.appendChild(span);
        return;
    }
    const name = document.createElement('span');
    name.className = 'mi-name';
    name.textContent = info.name ?? '未命名模型';
    dom.modelInfo.appendChild(name);

    const items = [
        ['hidden', info.hidden_size],
        ['inter', info.intermediate_size],
        ['vocab', info.vocab_size],
    ];
    const layers = document.createElement('label');
    layers.className = 'mi-chip mi-layer-count';
    const layersLabel = document.createElement('em');
    layersLabel.textContent = '层数';
    const layersInput = document.createElement('input');
    layersInput.type = 'number';
    layersInput.min = String(
        (info.structure?.prefix_layer_count ?? info.prefix_layer_count ?? 0)
        + (info.structure?.suffix_layer_count ?? info.suffix_layer_count ?? 0) + 1,
    );
    layersInput.step = '1';
    layersInput.value = String(info.layer_count ?? 1);
    layersInput.title = '总层数：循环 Pattern 会按此值展开，最后一轮可截断';
    const applyLayerCount = () => {
        if (!setModelLayerCount(layersInput.value)) {
            layersInput.value = String(AppState.flowchart?.model_info?.layer_count ?? info.layer_count);
            return;
        }
        renderModelInfo(AppState.flowchart?.model_info);
    };
    layersInput.addEventListener('change', applyLayerCount);
    layersInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            applyLayerCount();
            layersInput.blur();
        }
    });
    layers.append(layersLabel, layersInput);
    dom.modelInfo.appendChild(layers);
    const structure = info.structure;
    if (structure) {
        const summary = document.createElement('span');
        summary.className = 'mi-structure';
        summary.textContent = structure.pattern_cycle_length
            ? `前置 ${structure.prefix_layer_count} · 循环单元 ${structure.pattern_cycle_length} 层 · 循环重复 ${structure.full_cycle_count} 轮`
                + (structure.partial_cycle_layers ? ` +截断${structure.partial_cycle_layers}` : '')
                + (structure.suffix_layer_count ? ` · 尾部 ${structure.suffix_layer_count}` : '')
            : `前置 ${structure.prefix_layer_count} · 无循环`;
        summary.title = '总层数由前置段、循环段和尾部段共同展开；尾部段始终锚定模型末尾。';
        dom.modelInfo.appendChild(summary);
    }
    for (const [label, value] of items) {
        if (value == null) continue;
        const chip = document.createElement('span');
        chip.className = 'mi-chip';
        const k = document.createElement('em');
        k.textContent = label;
        const v = document.createElement('b');
        v.textContent = formatNumber(value);
        chip.append(k, v);
        dom.modelInfo.appendChild(chip);
    }
    const quality = info.metadata?.mapping_quality;
    if (quality) {
        const tag = document.createElement('span');
        tag.className = 'mi-quality';
        tag.textContent = MAPPING_QUALITY_TEXT[quality] ?? quality;
        tag.title = '模型映射质量；未建模项会在预设说明中列出。';
        dom.modelInfo.appendChild(tag);
    }
}

/* ---------------------------------------------------------------- 计算流程 */

/**
 * 流程图 + 左侧栏 → 完整 config。
 * 以当前完整 config 为底座，仅用左侧栏产出的四段覆盖，
 * 这样 schema_version、model.dtype/quantization 等界面未暴露的段不会丢失。
 */
async function buildConfig() {
    const base = { ...(AppState.baseConfig ?? {}), ...getSidebarConfig() };
    syncFlowchartModelInfo(base.model);
    const config = await flowchartToConfig(AppState.flowchart, base);
    AppState.baseConfig = config;
    return config;
}

function syncFlowchartModelInfo(model) {
    const info = AppState.flowchart?.model_info;
    if (!info || !model || typeof model !== 'object') return;
    const dimensions = model.dimensions ?? {};
    for (const key of ['layer_count', 'hidden_size', 'intermediate_size', 'vocab_size', 'padded_vocab_size']) {
        if (dimensions[key] != null) info[key] = Number(dimensions[key]);
    }
    if (model.name) info.name = String(model.name);
    if (model.id) info.model_id = String(model.id);
    if (model.dtype && typeof model.dtype === 'object') info.dtype = { ...model.dtype };
}

/** 「计算」按钮：组装 config → 估算 → 渲染结果 */
async function compute() {
    if (!AppState.flowchart) {
        toast('info', '请先加载一个模型预设');
        return;
    }
    try {
        await withLoading('求解中…', async () => {
            const config = await buildConfig();
            const results = await runEstimate(config);
            AppState.results = results;
            renderResults(results);
            setResultsCollapsed(false);
            markStale(false);
            Bus.emit(EVENTS.RESULTS_READY, results);
        });
        const feasible = AppState.results?.capacity?.capacity_feasible
            ?? AppState.results?.validity?.capacity_feasible;
        if (feasible === false) {
            toast('error', '估算完成：显存容量超出上限，性能为理论值');
        } else {
            toast('success', '估算完成');
        }
    } catch (error) {
        reportError(error, '估算失败');
    }
}

/** 导出当前 config */
async function exportConfig() {
    try {
        const config = await withLoading('生成配置…', buildConfig);
        const name = AppState.currentPresetId ?? AppState.flowchart?.model_info?.name ?? 'config';
        download(`${slug(name)}-config.json`, JSON.stringify(config, null, 2));
        toast('success', '配置已导出');
    } catch (error) {
        reportError(error, '导出失败');
    }
}

/** 导入 config 文件 */
async function importConfigFile(file) {
    if (!file) return;
    try {
        const text = await file.text();
        const config = JSON.parse(text);
        await withLoading('导入配置…', async () => {
            applyConfig(config);
            const flowchart = await configToFlowchart(config);
            renderFlowchart(flowchart);
            renderModelInfo(flowchart.model_info);
            renderOperatorPanel(null);
            AppState.selectedNode = null;
            AppState.currentPresetId = null;
            dom.presetDesc.textContent = `已导入 ${file.name}`;
            clearResults();
        });
        toast('success', `已导入 ${file.name}`);
    } catch (error) {
        if (error instanceof SyntaxError) {
            reportError(error, 'JSON 解析失败，请确认文件格式');
        } else {
            reportError(error, '导入失败');
        }
    }
}

function clearResults() {
    AppState.results = null;
    dom.resultsBody.innerHTML = '';
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    const glyph = document.createElement('div');
    glyph.className = 'empty-glyph';
    glyph.textContent = '∑';
    const text = document.createElement('p');
    text.textContent = '点击顶部「计算」按钮，将当前流程图与运行配置提交给求解器。';
    empty.append(glyph, text);
    dom.resultsBody.appendChild(empty);
    dom.resultsSummary.innerHTML = '';
    const hint = document.createElement('span');
    hint.className = 'hint';
    hint.textContent = '尚未计算';
    dom.resultsSummary.appendChild(hint);
    markStale(false);
}

/** 标记结果过期（流程图或配置变更后） */
function markStale(stale) {
    resultsStale = stale;
    dom.resultsPanel.classList.toggle('is-stale', stale && !!AppState.results);
    dom.btnCompute.classList.toggle('is-attention', stale);
}

/* ---------------------------------------------------------------- 事件绑定 */

function bindToolbar() {
    dom.presetSelect.addEventListener('change', () => {
        const id = dom.presetSelect.value;
        if (id) loadPreset(id).catch((error) => reportError(error, '预设加载失败'));
    });
    dom.btnCompute.addEventListener('click', compute);
    dom.btnExport.addEventListener('click', exportConfig);
    dom.btnImport.addEventListener('click', () => dom.fileImport.click());
    dom.fileImport.addEventListener('change', () => {
        importConfigFile(dom.fileImport.files?.[0]);
        dom.fileImport.value = '';
    });

    // 左侧栏折叠（同时收窄三栏网格的首列）
    const toggleSidebar = () => {
        const collapsed = dom.sidebar.classList.toggle('is-collapsed');
        dom.workspace.classList.toggle('is-sidebar-collapsed', collapsed);
        dom.sidebarToggle.textContent = collapsed ? '›' : '‹';
    };
    dom.sidebarToggle.addEventListener('click', toggleSidebar);
    dom.sidebarRail.addEventListener('click', toggleSidebar);

    // 结果面板折叠
    dom.resultsToggle.addEventListener('click', () => {
        setResultsCollapsed(!dom.resultsPanel.classList.contains('is-collapsed'));
    });

    // 画布工具
    document.getElementById('btn-zoom-in').addEventListener('click', () => zoomBy(1.2));
    document.getElementById('btn-zoom-out').addEventListener('click', () => zoomBy(1 / 1.2));
    document.getElementById('btn-fit').addEventListener('click', fitView);
    document.getElementById('btn-collapse-all').addEventListener('click', () => setAllGroupsCollapsed(true));
    document.getElementById('btn-expand-all').addEventListener('click', () => setAllGroupsCollapsed(false));
    dom.btnResetOffsets.addEventListener('click', () => {
        if (!resetNodeOffsets()) toast('info', '当前没有需要重置的节点微调偏移');
    });
    bindLayoutPanel();

    // 快捷键：Ctrl/Cmd + Enter 计算
    window.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            compute();
        }
    });
}

function setResultsCollapsed(collapsed) {
    dom.resultsPanel.classList.toggle('is-collapsed', collapsed);
    dom.resultsToggle.textContent = collapsed ? '⌃' : '⌄';
}

/* ------------------------------------------------ 布局参数面板 */

/** 弹层输入框同步为当前生效值（含已持久化的 overrides，铳制后的结果） */
function syncLayoutPanel() {
    const layout = getLinearLayout();
    for (const key of LAYOUT_PANEL_KEYS) {
        const input = document.getElementById(`layout-input-${key}`);
        if (input) input.value = String(layout[key]);
    }
}

function setLayoutPanelOpen(open) {
    dom.layoutPopover.hidden = !open;
    dom.btnLayout.classList.toggle('is-active', open);
    if (open) syncLayoutPanel();
}

function bindLayoutPanel() {
    dom.btnLayout.addEventListener('click', () => {
        setLayoutPanelOpen(dom.layoutPopover.hidden);
    });

    // 点击弹层与按钮之外的区域关闭；Esc 同样关闭
    document.addEventListener('pointerdown', (event) => {
        if (dom.layoutPopover.hidden) return;
        if (dom.layoutPopover.contains(event.target) || dom.btnLayout.contains(event.target)) return;
        setLayoutPanelOpen(false);
    });
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !dom.layoutPopover.hidden) setLayoutPanelOpen(false);
    });

    // 输入即生效：min/max 铳制 → 覆盖布局参数 → 持久化 → 重绘
    // 不强制 fitView，避免打断用户；需要时可点击“适应画布”
    for (const key of LAYOUT_PANEL_KEYS) {
        const input = document.getElementById(`layout-input-${key}`);
        if (!input) continue;
        input.addEventListener('input', () => {
            // valueAsNumber 对空输入返回 NaN（Number('') === 0 会把清空瞬间误判为 0）
            const value = input.valueAsNumber;
            if (!Number.isFinite(value)) return; // 输入中途（如空值）不应用
            const min = Number(input.min);
            const max = Number(input.max);
            applyLinearLayoutOverrides({ [key]: Math.min(max, Math.max(min, value)) });
            saveLinearLayout();
            renderFlowchart();
        });
    }

    dom.btnLayoutReset.addEventListener('click', () => {
        resetLinearLayout();
        try { localStorage.removeItem(LAYOUT_STORAGE_KEY); } catch { /* 忽略 */ }
        syncLayoutPanel();
        renderFlowchart();
        toast('success', '布局参数已恢复默认');
    });
}

function bindBus() {
    Bus.on(EVENTS.TOAST, ({ type, message }) => toast(type ?? 'info', message));
    Bus.on(EVENTS.FLOWCHART_CHANGED, () => {
        renderModelInfo(AppState.flowchart?.model_info);
        markStale(true);
    });
    Bus.on(EVENTS.CONFIG_CHANGED, ({ path } = {}) => {
        if (String(path ?? '').startsWith('model.')) {
            const model = getSidebarConfig().model;
            if (path === 'model.dimensions.layer_count') {
                const accepted = setModelLayerCount(model?.dimensions?.layer_count);
                if (!accepted && AppState.flowchart?.model_info) {
                    AppState.sidebarConfig.model.dimensions.layer_count = AppState.flowchart.model_info.layer_count;
                    loadFromConfig(getSidebarConfig());
                }
            } else {
                syncFlowchartModelInfo(model);
                downgradeOfficialMapping();
            }
            renderModelInfo(AppState.flowchart?.model_info);
        }
        markStale(true);
    });
    Bus.on(EVENTS.VIEW_CHANGED, ({ scale }) => {
        dom.zoomLevel.textContent = `${Math.round(scale * 100)}%`;
    });
}

function downgradeOfficialMapping() {
    const metadata = AppState.flowchart?.model_info?.metadata;
    if (!metadata || metadata.mapping_quality !== 'official_config_aligned') return;
    metadata.source_mapping_quality = metadata.mapping_quality;
    metadata.mapping_quality = 'customized_from_official';
}

function bindGlobalErrors() {
    window.addEventListener('unhandledrejection', (event) => {
        reportError(event.reason, '出现未处理的异常');
    });
    window.addEventListener('error', (event) => {
        if (event.error) reportError(event.error, '出现脚本错误');
    });
}

/* ---------------------------------------------------------------- UI 辅助 */

/** 包裹异步任务，展示遮罩 */
async function withLoading(label, task) {
    dom.loadingText.textContent = label;
    dom.loadingVeil.hidden = false;
    try {
        return await task();
    } finally {
        dom.loadingVeil.hidden = true;
    }
}

/** Toast 通知：success / error / info */
export function toast(type, message) {
    if (!dom.toastStack || !message) return;
    const item = document.createElement('div');
    item.className = `toast is-${type}`;
    const glyph = document.createElement('span');
    glyph.className = 'toast-glyph';
    glyph.textContent = type === 'success' ? '✓' : (type === 'error' ? '✕' : 'i');
    const text = document.createElement('p');
    text.textContent = message;
    const close = document.createElement('button');
    close.className = 'toast-close';
    close.textContent = '✕';

    const dismiss = () => {
        item.classList.add('is-leaving');
        setTimeout(() => item.remove(), 200);
    };
    close.addEventListener('click', dismiss);
    item.append(glyph, text, close);
    dom.toastStack.appendChild(item);
    setTimeout(dismiss, type === 'error' ? 8000 : 3600);
}

/** 统一错误上报：控制台留全量堆栈，界面给可读提示 */
function reportError(error, fallbackMessage) {
    console.error(fallbackMessage, error);
    let message = fallbackMessage;
    if (error instanceof ApiError) {
        message = `${fallbackMessage}：${error.message}`;
    } else if (error?.message) {
        message = `${fallbackMessage}：${error.message}`;
    }
    toast('error', message);
}

function download(filename, content) {
    const blob = new Blob([content], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function slug(value) {
    return String(value).trim().toLowerCase().replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'config';
}

function formatNumber(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return String(value);
    return num.toLocaleString('zh-CN');
}

/* ---------------------------------------------------------------- 入口 */

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { init(); });
} else {
    init();
}

export { init, compute, resultsStale };
