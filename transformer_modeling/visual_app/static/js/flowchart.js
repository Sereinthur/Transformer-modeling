/**
 * flowchart.js —— SVG 流程图引擎
 *
 * 职责：把流程图数据渲染为可交互的 SVG（节点 / 边 / Layer 组），
 *       并处理选中、算子替换请求、缩放、平移、组折叠展开等交互。
 */

import {
    AppState, Bus, EVENTS, findNodeById, walkNodes,
} from './api.js';
import {
    getBounds, isGroup, getLinearLayout, VIEW_PADDING,
} from './layout.js';
import { openInsertPicker } from './operator-panel.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** 视口状态：viewBox 左上角世界坐标 + 缩放比例 */
const view = { x: 0, y: 0, scale: 1 };
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 2.5;

let host = null;          // 宿主 DOM
let svg = null;
let gridRect = null;
let cycleLayer = null;    // 循环层段的外层摘要容器
let peripheralLayer = null;
let groupLayer = null;    // Layer 组底板
let edgeLayer = null;     // 连线
let nodeLayer = null;     // 节点（含组内子节点）
let hostSize = { w: 1000, h: 700 };
let currentLayout = {};
let firstRender = true;

/** nodeId → 节点 <g> 元素 */
const nodeEls = new Map();

/** 新建层段的初始算子序列。创建后每个节点都是独立配置，不存在共享槽位。 */
const STAGE_CHILDREN = [
    ['norm_pre', 'norm', 'rms_norm', 'Attention 前 Norm'],
    ['attention', 'attention', 'standard_attention', 'Attention'],
    ['residual_1', 'residual', 'standard_residual', 'Attention 残差'],
    ['norm_post', 'norm', 'rms_norm', 'FFN 前 Norm'],
    ['ffn', 'ffn', 'gated_ffn', 'FFN'],
    ['residual_2', 'residual', 'standard_residual', 'FFN 残差'],
];

/* ---------------------------------------------------------------- 分类映射 */

/** slot / type → 视觉分类（与 CSS 中的 cat-* 配色一一对应） */
const SLOT_CATEGORY = {
    embedding: 'embedding',
    attention: 'attention',
    ffn: 'ffn',
    moe: 'ffn',
    norm: 'norm',
    residual: 'residual',
    output: 'output',
    layer: 'layer',
};

/** 判断节点视觉分类 */
export function categoryOf(node) {
    if (!node) return 'default';
    if (node.type === 'layer_group') return 'layer';
    const bySlot = SLOT_CATEGORY[node.slot];
    if (bySlot) return bySlot;
    const type = String(node.type ?? '');
    if (type.includes('embedding')) return 'embedding';
    if (type.includes('attention')) return 'attention';
    if (type.includes('ffn') || type.includes('moe') || type.includes('mlp')) return 'ffn';
    if (type.includes('norm')) return 'norm';
    if (type.includes('residual')) return 'residual';
    if (type.includes('head') || type.includes('sampling') || type.includes('output')) return 'output';
    return 'default';
}

/* ---------------------------------------------------------------- 初始化 */

/**
 * 初始化画布。
 * @param {HTMLElement} container 宿主容器
 */
export function initFlowchart(container) {
    host = container;
    host.innerHTML = '';

    svg = el('svg', { class: 'fc-svg' });
    svg.appendChild(buildDefs());

    const viewportGroup = el('g', { class: 'fc-viewport' });
    gridRect = el('rect', { class: 'fc-grid', x: -4000, y: -4000, width: 12000, height: 12000, fill: 'url(#fc-grid-pattern)' });
    cycleLayer = el('g', { class: 'fc-layer-cycles' });
    peripheralLayer = el('g', { class: 'fc-layer-peripherals' });
    groupLayer = el('g', { class: 'fc-layer-groups' });
    edgeLayer = el('g', { class: 'fc-layer-edges' });
    nodeLayer = el('g', { class: 'fc-layer-nodes' });
    viewportGroup.append(
        gridRect, cycleLayer, peripheralLayer,
        groupLayer, edgeLayer, nodeLayer,
    );
    svg.appendChild(viewportGroup);
    host.appendChild(svg);

    measureHost();
    bindInteractions();
    ensureFontRelayout();

    if (typeof ResizeObserver === 'function') {
        new ResizeObserver(() => {
            measureHost();
            applyView();
        }).observe(host);
    } else {
        window.addEventListener('resize', () => { measureHost(); applyView(); });
    }
    applyView();
}

/** defs：分类渐变、箭头标记、阴影滤镜、网格图案 */
function buildDefs() {
    const defs = el('defs');

    // 每个分类一套自上而下的渐变（颜色取自 CSS 变量的镜像值，SVG 内需字面量）
    const palette = {
        embedding: ['#5fb0ff', '#2f7ede'],
        attention: ['#ffa063', '#e5701f'],
        ffn: ['#63d9d1', '#2ba9a0'],
        norm: ['#aab7b8', '#7c8a8c'],
        residual: ['#bd7cf9', '#8b34e0'],
        output: ['#f76a6a', '#cf2c2c'],
        layer: ['#7c7ff5', '#4a4dd1'],
        default: ['#8b93a7', '#5d6479'],
    };
    for (const [cat, [from, to]] of Object.entries(palette)) {
        const grad = el('linearGradient', { id: `fc-grad-${cat}`, x1: '0', y1: '0', x2: '0.35', y2: '1' });
        grad.append(
            el('stop', { offset: '0%', 'stop-color': from }),
            el('stop', { offset: '100%', 'stop-color': to }),
        );
        defs.appendChild(grad);
    }

    // 箭头
    for (const [id, cls, size] of [['fc-arrow', 'fc-arrow-head', 7], ['fc-arrow-inner', 'fc-arrow-head is-inner', 5.5]]) {
        const marker = el('marker', {
            id, viewBox: '0 0 10 10', refX: '8.5', refY: '5',
            markerWidth: String(size), markerHeight: String(size), orient: 'auto-start-reverse',
        });
        marker.appendChild(el('path', { class: cls, d: 'M 1 1 L 9 5 L 1 9 z' }));
        defs.appendChild(marker);
    }

    // 节点投影
    const shadow = el('filter', { id: 'fc-shadow', x: '-30%', y: '-30%', width: '160%', height: '180%' });
    shadow.append(
        el('feDropShadow', { dx: '0', dy: '3', stdDeviation: '4', 'flood-color': '#05050c', 'flood-opacity': '0.55' }),
    );
    defs.appendChild(shadow);

    // 选中光晕
    const glow = el('filter', { id: 'fc-glow', x: '-40%', y: '-40%', width: '180%', height: '200%' });
    glow.append(
        el('feDropShadow', { dx: '0', dy: '0', stdDeviation: '6', 'flood-color': '#7dd3fc', 'flood-opacity': '0.85' }),
    );
    defs.appendChild(glow);

    // 蓝图网格
    const pattern = el('pattern', { id: 'fc-grid-pattern', width: '28', height: '28', patternUnits: 'userSpaceOnUse' });
    pattern.append(
        el('path', { class: 'fc-grid-line', d: 'M 28 0 L 0 0 0 28' }),
        el('circle', { class: 'fc-grid-dot', cx: '0', cy: '0', r: '1' }),
    );
    defs.appendChild(pattern);

    return defs;
}

/* ---------------------------------------------------------------- 渲染主流程 */

/**
 * 渲染完整流程图。
 * @param {object} [flowchartData] 不传则重绘 AppState.flowchart
 */
export function renderFlowchart(flowchartData) {
    if (flowchartData) {
        AppState.flowchart = flowchartData;
        firstRender = true;
        nodeEls.forEach((element) => element.remove());
        nodeEls.clear();
    }
    const data = AppState.flowchart;
    if (!svg || !data) return;

    refreshStructureMetadata(data);
    currentLayout = {};
    cycleLayer.innerHTML = '';
    peripheralLayer.innerHTML = '';
    edgeLayer.innerHTML = ''; // 连线每次全量重建（组内连线由组渲染阶段追加）
    groupLayer.innerHTML = '';
    nodeLayer.innerHTML = '';
    nodeEls.clear();
    renderLinearBackboneScene(data);

    // 清理已不存在的节点
    syncSelection();

    if (firstRender) {
        firstRender = false;
        fitView();
    } else {
        // A redraw after an edit must preserve the user's zoom and pan.
        applyView();
    }
}

/* ---------------------------------------------------------------- 线性主干渲染 */

function shortArchitectureName(node) {
    const names = { kda: 'KDA', gated_mla: 'Gated MLA', dsa_attention: 'DSA', csa_attention: 'CSA', hca_attention: 'HCA', sliding_window_attention: 'Sliding Attention', standard_attention: 'Attention', moe: 'MoE', gated_ffn: 'Dense FFN', dense_ffn: 'Dense FFN' };
    return names[node?.type] ?? node?.label ?? node?.type ?? '—';
}

/* 画布严格按 Schema v3 operations 顺序渲染；残差、mHC 与 AttnRes
   和 Attention/MoE 一样，都是可选中、可移动、可替换的真实算子卡。 */
function renderLinearBackboneScene(data) {
    const nodes = data.nodes ?? [];
    const groups = nodes.filter(isGroup);
    const embedding = nodes.find((node) => node.role === 'embedding') ?? nodes[0];
    const outputNorm = nodes.find((node) => node.role === 'output_norm');
    const outputHead = nodes.find((node) => node.role === 'output_head');
    const outputSampling = nodes.find((node) => node.role === 'output_sampling');
    const metrics = getLinearLayout();
    const width = metrics.canvasW;
    const centerX = metrics.centerX;
    const cardW = metrics.cardW;
    const cardH = metrics.cardH;
    const cardX = centerX - cardW / 2;
    const gap = metrics.cardGap;
    const groupGap = metrics.groupGap;
    const positions = new Map();
    const groupBounds = [];
    let y = metrics.topY;
    const endpoints = [embedding, ...groups.flatMap((group) => group.collapsed ? [] : (group.children ?? [])), outputNorm, outputHead, outputSampling].filter(Boolean);
    const estimatedHeight = 100 + endpoints.length * (cardH + gap) + groups.length * groupGap;
    currentLayout = { __scene: { x: 0, y: 0, width, height: Math.max(620, estimatedHeight) } };
    sceneText(peripheralLayer, 'Ordered backbone — select an operator to edit; double-click to replace', centerX, 26, 'fc-linear-title', { 'text-anchor': 'middle' });

    // 用户拖拽微调产生的垂直偏移（_offset_y，运行时字段，提交后端前被剥离）：
    // 仅平移单张卡片，不推移布局游标 y，保持线性主干语义；
    // 卡片的 positions 记录含偏移，主干脊柱线按实际端口坐标连接。
    const offsetYOf = (node) => Number(node._offset_y) || 0;
    const drawCard = (node, label, sub = '', height = cardH, actions = null) => {
        const box = linearNode(node, cardX, y + offsetYOf(node), cardW, height, label, sub, actions);
        positions.set(node.id, { input: box.y, output: box.y + box.height, box });
        y += height + gap;
        return box;
    };
    if (embedding) drawCard(embedding, embedding.label ?? 'Embedding', embedding.type ?? '');
    let lastMain = embedding;
    const spineTo = (topY) => {
        if (!lastMain) return;
        const from = positions.get(lastMain.id)?.output;
        if (from != null) scenePath(edgeLayer, `M ${centerX} ${from} V ${topY}`, 'fc-linear-main');
    };
    for (const group of groups) {
        const startY = y - 8;
        const stage = obtainNodeEl(group.id, groupLayer);
        stage.setAttribute('class', `fc-linear-stage is-${group.group_kind}${AppState.selectedNode?.id === group.id ? ' is-selected' : ''}`);
        stage.dataset.nodeId = group.id;
        stage.dataset.kind = 'group';
        nodeEls.set(group.id, stage);
        const children = group.collapsed ? [] : (group.children ?? []);
        for (const child of children) {
            spineTo(y + offsetYOf(child));
            const summary = linearOperatorSummary(child);
            drawCard(child, child.label ?? shortArchitectureName(child), summary,
                cardH, buildOperatorActions(group, child));
            lastMain = child;
        }
        if (!children.length) {
            sceneText(stage, 'Collapsed layer segment', centerX, y + 7, 'fc-linear-stage-note', { 'text-anchor': 'middle' });
            y += 24;
        }
        const endY = y - gap + 8;
        const bounds = { x: metrics.frameX, y: startY, width: metrics.frameW, height: Math.max(54, endY - startY), group };
        currentLayout[group.id] = bounds;
        place(stage, 0, 0);
        stage.append(
            el('rect', { class: 'fc-linear-stage-hit', x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height, rx: 8, ry: 8 }),
            el('path', { class: 'fc-linear-bracket', d: `M ${bounds.x + 14} ${bounds.y} H ${bounds.x} V ${bounds.y + bounds.height} H ${bounds.x + 14}` }),
            text(linearStageLabel(group), { class: 'fc-linear-stage-label', x: bounds.x - 8, y: bounds.y + 12, 'text-anchor': 'end' }),
            // 重复次数标签放在组框顶部上方（groupGap 走廊只有脊柱线，靠 CSS halo 保持可读）
            text(linearStageRepeat(group), { class: 'fc-linear-stage-repeat', x: bounds.x + bounds.width, y: bounds.y - 6, 'text-anchor': 'end' }),
            // 组级「+ 添加算子」：放在阶段标签下方（组框左外侧），避开右上角 ×N 残差角标；折叠态保留
            buildStageAddButton(bounds),
        );
        groupBounds.push(bounds);
        y += groupGap;
    }
    for (const endpoint of [outputNorm, outputHead, outputSampling]) {
        if (!endpoint) continue;
        spineTo(y + offsetYOf(endpoint));
        drawCard(endpoint, endpoint.label ?? shortArchitectureName(endpoint), endpoint.type ?? '');
        lastMain = endpoint;
    }
    // 不能只依赖预估高度：卡片高度会随布局设置变化，且层组会额外占用
    // groupGap。若 scene 高度偏小，fitView 会把底部层段裁到 SVG 可视区之外，
    // 使循环段既看不到也无法点击编辑。
    currentLayout.__scene.height = Math.max(620, y + metrics.topY);
    groupBounds.forEach((bounds) => cycleLayer.appendChild(el('rect', {
        class: `fc-linear-stage-bg is-${bounds.group.group_kind}`,
        x: bounds.x + 22, y: bounds.y - 7, width: bounds.width - 26, height: bounds.height + 14, rx: 8, ry: 8,
    })));

    // 节点入 DOM 后统一测量截断，避免标签溢出卡片
    fitLinearNodeLabels();

}

function linearNode(node, x, y, width, height, label, sub, actions = null) {
    const group = obtainNodeEl(node.id, nodeLayer);
    const category = categoryOf(node);
    group.setAttribute('class', `fc-linear-node cat-${category}${AppState.selectedNode?.id === node.id ? ' is-selected' : ''}`);
    group.dataset.nodeId = node.id;
    group.dataset.kind = 'node';
    nodeEls.set(node.id, group);
    currentLayout[node.id] = { x, y, width, height, kind: 'linear' };
    place(group, x, y); group.innerHTML = '';
    const compact = height < 36; // 内联残差矮卡：文字按卡高比例排布，避免溢出
    const labelY = sub ? (compact ? Math.round(height * 0.36) : 17) : (compact ? height / 2 : 22);
    const subY = compact ? Math.round(height * 0.74) : 32;
    group.append(
        el('rect', { class: 'fc-linear-node-bg', x: 0, y: 0, width, height, rx: 7, ry: 7 }),
        el('circle', { class: 'fc-linear-port', cx: width / 2, cy: 0, r: 3 }),
        el('circle', { class: 'fc-linear-port', cx: width / 2, cy: height, r: 3 }),
        text(label, { class: 'fc-linear-node-label', x: width / 2, y: labelY, 'text-anchor': 'middle', 'dominant-baseline': 'central' }),
    );
    if (sub) group.appendChild(text(sub, { class: 'fc-linear-node-sub', x: width / 2, y: subY, 'text-anchor': 'middle', 'dominant-baseline': 'central' }));
    if (actions?.length) group.appendChild(buildActionsBar(actions, width));
    return { x, y, width, height };
}

/* ---------------- 卡片悬浮操作条（增/删/移/替换直接入口） ---------------- */

const ACTION_BTN_SIZE = 15;
const ACTION_BTN_GAP = 4;

/**
 * 层内算子卡的操作条按钮矩阵：↑ 上移、↓ 下移、⇄ 替换、+ 在其后插入、× 删除。
 * 组内最后一个算子不可删除；其它算子均可独立移动、替换或删除。
 */
function buildOperatorActions(group, node) {
    const onlyChild = (group.children ?? []).length <= 1;
    const removeTitle = onlyChild ? '每组至少保留一个算子' : '删除算子';
    return [
        { action: 'up', glyph: '↑', title: '上移' },
        { action: 'down', glyph: '↓', title: '下移' },
        { action: 'replace', glyph: '⇄', title: '替换算子' },
        { action: 'insert', glyph: '+', title: '在此后插入算子' },
        { action: 'remove', glyph: '×', title: removeTitle, disabled: onlyChild },
    ];
}

/** 悬浮操作条 <g class="fc-node-actions">：右对齐悬浮在卡片上沿，CSS hover 显示 */
function buildActionsBar(actions, cardWidth) {
    const bar = el('g', { class: 'fc-node-actions' });
    const total = actions.length * ACTION_BTN_SIZE + (actions.length - 1) * ACTION_BTN_GAP;
    let bx = cardWidth - 4 - total;
    for (const spec of actions) {
        bar.appendChild(actionButton(spec, bx, -ACTION_BTN_SIZE / 2 - 1));
        bx += ACTION_BTN_SIZE + ACTION_BTN_GAP;
    }
    return bar;
}

/** 组级「+ 添加算子」按钮：阶段标签下方，折叠态保留；点击在组末尾插入 */
function buildStageAddButton(bounds) {
    return actionButton(
        { action: 'add-op', glyph: '+', title: '在该组末尾添加算子' },
        bounds.x - 8 - ACTION_BTN_SIZE, bounds.y + 20, 'fc-stage-add',
    );
}

/** 单个操作按钮：圆角小方块 + 符号 + <title> 提示；置灰态附 is-disabled 类 */
function actionButton(spec, x, y, extraClass = '') {
    const btn = el('g', {
        class: `fc-node-action${spec.disabled ? ' is-disabled' : ''}${extraClass ? ` ${extraClass}` : ''}`,
        'data-action': spec.action,
        transform: `translate(${x}, ${y})`,
    });
    btn.appendChild(el('rect', {
        class: 'fc-node-action-bg',
        x: 0, y: 0, width: ACTION_BTN_SIZE, height: ACTION_BTN_SIZE, rx: 3.5, ry: 3.5,
    }));
    btn.appendChild(text(spec.glyph, {
        class: 'fc-node-action-glyph',
        x: ACTION_BTN_SIZE / 2, y: ACTION_BTN_SIZE / 2 + 0.5,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
    }));
    const tip = el('title');
    tip.textContent = spec.title;
    btn.appendChild(tip);
    return btn;
}

function linearOperatorSummary(node) {
    if (node.type !== 'moe') return node.type ?? '';
    const p = node.params ?? {};
    const experts = p.routed_expert_count ?? p.expert_count ?? p.num_experts;
    const topK = p.experts_per_token ?? p.top_k;
    const shared = p.shared_expert_count ?? p.shared_experts;
    return [experts && `E${experts}`, topK && `top-${topK}`, shared && `shared ${shared}`].filter(Boolean).join(' · ') || 'MoE';
}

function linearStageLabel(group) {
    const stats = group._structure ?? {};
    const range = stats.first_layer ? `L${stats.first_layer}${stats.last_layer && stats.last_layer !== stats.first_layer ? `–${stats.last_layer}` : ''}` : 'layer';
    return `${group.group_kind ?? 'pattern'} · ${range}`;
}

function linearStageRepeat(group) {
    const stats = group._structure ?? {};
    return group.group_kind === 'pattern'
        ? `×${group.repeat ?? 1} · appears ${stats.occurrences ?? 0}`
        : `×${group.repeat ?? 1}`;
}

/**
 * 渲染完成后的标签测量 pass：超宽标签逐字符截断加 '…'，
 * 全文存入 <title> 供悬浮显示。按各卡片实际宽度单独判定（内联残差矮卡更窄）。
 */
function fitLinearNodeLabels() {
    if (!svg) return;
    svg.querySelectorAll('.fc-linear-node-label, .fc-linear-node-sub').forEach((labelEl) => {
        const full = labelEl.textContent;
        if (!full) return;
        const bg = labelEl.parentNode?.querySelector('.fc-linear-node-bg');
        const maxWidth = (bg ? Number(bg.getAttribute('width')) : 220) - 16;
        const measured = () => {
            try { return labelEl.getComputedTextLength(); } catch { return 0; }
        };
        if (measured() <= maxWidth) return;
        let str = full;
        while (str.length > 1) {
            str = str.slice(0, -1);
            labelEl.textContent = `${str}…`;
            if (measured() <= maxWidth) break;
        }
        const title = el('title');
        title.textContent = full;
        labelEl.appendChild(title);
    });
}

function sceneText(layer, value, x, y, className = 'fc-scene-caption', attrs = {}) {
    layer.appendChild(text(value, { class: className, x, y, ...attrs }));
}

function scenePath(layer, d, kind = '', arrow = true) {
    layer.appendChild(el('path', { class: `fc-scene-edge ${kind}`, d, ...(arrow ? { 'marker-end': 'url(#fc-arrow)' } : {}) }));
}

/* ---------------------------------------------------------------- 对外操作 */

/** 选中真实算子或层组节点（传 null 取消选中）。 */
export function selectNode(nodeId) {
    const hit = nodeId ? findNodeById(AppState.flowchart, nodeId) : null;
    AppState.selectedNode = hit ? hit.node : null;
    syncSelection();
    Bus.emit(EVENTS.NODE_SELECTED, AppState.selectedNode);
}

/** 同步选中态样式 */
function syncSelection() {
    const selectedId = AppState.selectedNode?.id ?? null;
    for (const [id, element] of nodeEls) {
        element.classList.toggle('is-selected', id === selectedId);
    }
}

/**
 * 替换算子类型：保留同名兼容参数，缺失参数取新算子默认值。
 */
export function updateNodeType(nodeId, newType) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit) return null;
    const { node } = hit;
    const schema = AppState.operatorSchemas?.[newType];
    const oldParams = node.params ?? {};
    const nextParams = {};

    for (const [key, spec] of Object.entries(schema?.params ?? {})) {
        const previous = oldParams[key];
        nextParams[key] = previous !== undefined && previous !== null
            ? previous : defaultParam(newType, key, spec);
    }

    const impls = schema?.implementations ?? [];
    node.type = newType;
    node.slot = schema?.slot ?? 'any';
    node.label = schema?.chinese_name ?? newType;
    node.params = { ...nextParams };
    node.implementation = impls.length && impls.includes(node.implementation)
        ? node.implementation : (impls[0] ?? 'default');

    markArchitectureCustomized();
    renderFlowchart();
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: 'type', nodeId });
    Bus.emit(EVENTS.NODE_SELECTED, node);
    return node;
}

/** 更新独立算子参数（增量合并）。 */
export function updateNodeParams(nodeId, params) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit) return null;
    const { node } = hit;
    node.params = { ...(node.params ?? {}), ...params };
    markArchitectureCustomized();
    renderFlowchart();
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: 'params', nodeId });
    return node;
}

/** 更新节点顶层字段（如 implementation、repeat）。 */
export function updateNodeField(nodeId, field, value) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit) return null;
    hit.node[field] = value;
    markArchitectureCustomized();
    renderFlowchart();
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: field, nodeId });
    return hit.node;
}


/** 更新模型总层数；前置层后至少保留一个循环层，保证配置仍有有效 Pattern。 */
export function setModelLayerCount(value) {
    const flowchart = AppState.flowchart;
    const next = Math.round(Number(value));
    if (!flowchart || !Number.isFinite(next)) return false;
    const prefixCount = (flowchart.nodes ?? [])
        .filter((node) => isGroup(node) && node.group_kind === 'prefix')
        .reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const suffixCount = (flowchart.nodes ?? [])
        .filter((node) => isGroup(node) && node.group_kind === 'suffix')
        .reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const minimum = prefixCount + suffixCount + 1;
    if (next < minimum) {
        Bus.emit(EVENTS.TOAST, {
            type: 'info', message: `总层数至少为 ${minimum}，以保留一个循环 Pattern 层。`,
        });
        return false;
    }
    const selectedId = AppState.selectedNode?.id ?? null;
    flowchart.model_info = { ...(flowchart.model_info ?? {}), layer_count: next };
    markArchitectureCustomized();
    refreshStructureMetadata(flowchart);
    renderFlowchart();
    if (selectedId) selectNode(selectedId);
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: 'layer_count' });
    return true;
}

/** Update global model metadata that is serialized with the flowchart. */
export function updateModelInfo(updates) {
    const flowchart = AppState.flowchart;
    if (!flowchart || !updates || typeof updates !== 'object') return false;
    flowchart.model_info = { ...(flowchart.model_info ?? {}), ...updates };
    markArchitectureCustomized();
    renderFlowchart();
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: 'model_info' });
    return true;
}

/**
 * 设置循环单元的完整重复次数。它对应的是 V4 的「[HCA → CSA] ×29」
 * 或 K3 的「[KDA ×3 → MLA] ×22」，而不是某个算子在单元内的 ×N。
 * 以完整循环为单位改写总层数，避免把侧边的 Pattern 层号范围误当作只读标签。
 */
export function setPatternCycleCount(value) {
    const flowchart = AppState.flowchart;
    const cycles = Math.round(Number(value));
    if (!flowchart || !Number.isFinite(cycles) || cycles < 1) return false;
    const groups = (flowchart.nodes ?? []).filter(isGroup);
    const prefixCount = groups.filter((node) => node.group_kind === 'prefix')
        .reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const cycleLength = groups.filter((node) => node.group_kind === 'pattern')
        .reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const suffixCount = groups.filter((node) => node.group_kind === 'suffix')
        .reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    if (!cycleLength) return false;
    const selectedId = AppState.selectedNode?.id ?? null;
    flowchart.model_info = {
        ...(flowchart.model_info ?? {}),
        layer_count: prefixCount + cycleLength * cycles + suffixCount,
    };
    markArchitectureCustomized();
    refreshStructureMetadata(flowchart);
    renderFlowchart();
    if (selectedId) selectNode(selectedId);
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason: 'pattern_cycle_count' });
    return true;
}

/** 同后端 model_info 对齐地重算循环统计；用于总层数与阶段结构的即时编辑。 */
function refreshStructureMetadata(flowchart) {
    const nodes = flowchart?.nodes ?? [];
    const prefixes = nodes.filter((node) => isGroup(node) && node.group_kind === 'prefix');
    const patterns = nodes.filter((node) => isGroup(node) && node.group_kind === 'pattern');
    const suffixes = nodes.filter((node) => isGroup(node) && node.group_kind === 'suffix');
    const info = flowchart.model_info ?? (flowchart.model_info = {});
    const layerCount = Math.max(0, Number(info.layer_count ?? 0));
    const prefixCount = prefixes.reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const cycleLength = patterns.reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const suffixCount = suffixes.reduce((sum, node) => sum + Math.max(1, Number(node.repeat ?? 1)), 0);
    const remaining = Math.max(0, layerCount - prefixCount - suffixCount);
    const fullCycles = cycleLength ? Math.floor(remaining / cycleLength) : 0;
    const remainder = cycleLength ? remaining % cycleLength : 0;
    info.prefix_layer_count = prefixCount;
    info.pattern_cycle_length = cycleLength;
    info.suffix_layer_count = suffixCount;
    info.structure = {
        prefix_layer_count: prefixCount,
        pattern_cycle_length: cycleLength,
        full_cycle_count: fullCycles,
        partial_cycle_layers: remainder,
        suffix_layer_count: suffixCount,
    };

    let cursor = 0;
    for (const node of prefixes) {
        const repeat = Math.max(1, Number(node.repeat ?? 1));
        node._structure = {
            kind: 'prefix', occurrences: repeat, first_layer: cursor + 1,
            last_layer: cursor + repeat, repeat_per_cycle: null,
        };
        cursor += repeat;
    }
    let offset = 0;
    for (const node of patterns) {
        const repeat = Math.max(1, Number(node.repeat ?? 1));
        const partial = Math.max(0, Math.min(repeat, remainder - offset));
        const occurrences = fullCycles * repeat + partial;
        const lastLayer = !occurrences ? 0 : (partial
            ? prefixCount + fullCycles * cycleLength + offset + partial
            : prefixCount + (fullCycles - 1) * cycleLength + offset + repeat);
        node._structure = {
            kind: 'pattern', occurrences, first_layer: prefixCount + offset + 1,
            last_layer: lastLayer, repeat_per_cycle: repeat,
            cycle_position_start: offset + 1, cycle_position_end: offset + repeat,
            full_cycle_count: fullCycles, partial_occurrences: partial,
        };
        offset += repeat;
    }
    let suffixCursor = layerCount - suffixCount;
    for (const node of suffixes) {
        const repeat = Math.max(1, Number(node.repeat ?? 1));
        node._structure = {
            kind: 'suffix', occurrences: repeat, first_layer: suffixCursor + 1,
            last_layer: suffixCursor + repeat, repeat_per_cycle: null,
        };
        suffixCursor += repeat;
    }
}

export function insertLayerOperation(groupId, afterOperationId = null, type = 'unmodeled') {
    const group = (AppState.flowchart?.nodes ?? []).find((node) => node.id === groupId && isGroup(node));
    if (!group) return null;
    const schema = AppState.operatorSchemas?.[type] ?? {};
    const ids = new Set((group.children ?? []).map((node) => node.operation_id ?? node.role));
    let index = 1;
    while (ids.has(`op_${index}`)) index += 1;
    const operationId = `op_${index}`;
    const child = {
        id: `${group.id}_${operationId}`, operation_id: operationId, role: operationId,
        type, slot: schema.slot ?? 'any', label: schema.chinese_name ?? type,
        implementation: schema.implementations?.[0] ?? 'default', params: schemaDefaults(type),
    };
    const children = group.children ?? (group.children = []);
    const anchorIndex = children.findIndex((node) => (node.operation_id ?? node.role) === afterOperationId);
    // 锚点不存在时兜底追加到组末尾，避免 findIndex 返回 -1 导致静默插入组首
    const position = afterOperationId == null || anchorIndex < 0 ? children.length : anchorIndex + 1;
    children.splice(position, 0, child);
    commitStructureChange('insert_operation', child.id);
    return child;
}

export function moveLayerOperation(nodeId, direction) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit?.parent || !['up', 'down'].includes(direction)) return false;
    const children = hit.parent.children ?? [];
    const index = children.indexOf(hit.node);
    const target = index + (direction === 'up' ? -1 : 1);
    if (index < 0 || target < 0 || target >= children.length) return false;
    [children[index], children[target]] = [children[target], children[index]];
    commitStructureChange('move_operation', nodeId);
    return true;
}

export function removeLayerOperation(nodeId) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit?.parent) return false;
    if ((hit.parent.children ?? []).length <= 1) return false;
    hit.parent.children.splice(hit.parent.children.indexOf(hit.node), 1);
    AppState.selectedNode = null;
    commitStructureChange('remove_operation');
    return true;
}

/** 在指定阶段后插入默认阶段；阶段类型跟随插入位置。 */
export function addLayerGroup(afterId = null) {
    const flowchart = AppState.flowchart;
    if (!flowchart) return null;
    const nodes = flowchart.nodes ?? [];
    const source = afterId ? nodes.find((item) => item.id === afterId && isGroup(item)) : null;
    const kind = source?.group_kind ?? 'pattern';
    const anchor = source ?? [...nodes].reverse().find((item) => isGroup(item) && item.group_kind === kind);
    const group = createLayerGroup(uniqueGroupId(kind), kind, anchor);
    const outputIndex = nodes.findIndex((item) => item.role === 'output_norm');
    nodes.splice(anchor ? nodes.indexOf(anchor) + 1 : outputIndex, 0, group);
    commitStructureChange('add', group.id);
    return group;
}

/** 复制一个阶段（包括当前的算子类型与参数）。 */
export function duplicateLayerGroup(nodeId) {
    const flowchart = AppState.flowchart;
    const source = flowchart?.nodes?.find((item) => item.id === nodeId && isGroup(item));
    if (!source) return null;
    const groupId = uniqueGroupId(source.group_kind);
    const copy = JSON.parse(JSON.stringify(source));
    copy.id = groupId;
    copy.collapsed = false;
    copy.children = (copy.children ?? []).map((child) => ({ ...child, id: `${groupId}_${child.role}` }));
    flowchart.nodes.splice(flowchart.nodes.indexOf(source) + 1, 0, copy);
    commitStructureChange('duplicate', copy.id);
    return copy;
}

/** 删除阶段；至少保留一个循环 Pattern，保证可转换为有效模型配置。 */
export function removeLayerGroup(nodeId) {
    const flowchart = AppState.flowchart;
    const group = flowchart?.nodes?.find((item) => item.id === nodeId && isGroup(item));
    if (!group) return false;
    const patterns = flowchart.nodes.filter((item) => isGroup(item) && item.group_kind === 'pattern');
    if (group.group_kind === 'pattern' && patterns.length <= 1) {
        Bus.emit(EVENTS.TOAST, { type: 'info', message: '至少需要保留一个循环 Pattern 阶段。' });
        return false;
    }
    flowchart.nodes.splice(flowchart.nodes.indexOf(group), 1);
    if (AppState.selectedNode?.id === group.id || AppState.selectedNode?.id?.startsWith(`${group.id}_`)) {
        AppState.selectedNode = null;
        Bus.emit(EVENTS.NODE_SELECTED, null);
    }
    commitStructureChange('delete');
    return true;
}

/** 仅在同类阶段内调整顺序，前置层永远位于循环 Pattern 之前。 */
export function moveLayerGroup(nodeId, direction) {
    const flowchart = AppState.flowchart;
    const group = flowchart?.nodes?.find((item) => item.id === nodeId && isGroup(item));
    if (!group || !['up', 'down'].includes(direction)) return false;
    const sameKind = flowchart.nodes.filter((item) => isGroup(item) && item.group_kind === group.group_kind);
    const current = sameKind.indexOf(group);
    const target = current + (direction === 'up' ? -1 : 1);
    if (target < 0 || target >= sameKind.length) return false;
    const other = sameKind[target];
    const left = flowchart.nodes.indexOf(group);
    const right = flowchart.nodes.indexOf(other);
    [flowchart.nodes[left], flowchart.nodes[right]] = [flowchart.nodes[right], flowchart.nodes[left]];
    commitStructureChange('move', group.id);
    return true;
}

/** 切换“前置层 / 循环 Pattern”语义，并阻止改走最后一个循环 Pattern。 */
export function setLayerGroupKind(nodeId, kind) {
    const flowchart = AppState.flowchart;
    const group = flowchart?.nodes?.find((item) => item.id === nodeId && isGroup(item));
    if (!group || !['prefix', 'pattern', 'suffix'].includes(kind) || group.group_kind === kind) return false;
    const patterns = flowchart.nodes.filter((item) => isGroup(item) && item.group_kind === 'pattern');
    if (group.group_kind === 'pattern' && kind !== 'pattern' && patterns.length <= 1) {
        Bus.emit(EVENTS.TOAST, { type: 'info', message: '至少需要保留一个循环 Pattern 阶段。' });
        return false;
    }
    group.group_kind = kind;
    commitStructureChange('kind', group.id);
    return true;
}

function createLayerGroup(groupId, kind, template = null) {
    if (template?.children?.length) {
        return {
            id: groupId, type: 'layer_group', slot: 'layer',
            label: stageKindLabel(kind), repeat: 1, group_kind: kind,
            children: template.children.map((child) => ({
                ...JSON.parse(JSON.stringify(child)), id: `${groupId}_${child.operation_id ?? child.role}`,
            })),
        };
    }
    return {
        id: groupId, type: 'layer_group', slot: 'layer',
        label: stageKindLabel(kind), repeat: 1, group_kind: kind,
        children: STAGE_CHILDREN
          .map(([role, slot, type, label]) => ({
            id: `${groupId}_${role}`, operation_id: role, type, slot, role,
            label: AppState.operatorSchemas?.[type]?.chinese_name ?? label,
            implementation: AppState.operatorSchemas?.[type]?.implementations?.[0] ?? 'default',
            params: schemaDefaults(type),
        })),
    };
}

function schemaDefaults(type) {
    const params = AppState.operatorSchemas?.[type]?.params ?? {};
    return Object.fromEntries(Object.entries(params)
        .map(([key, spec]) => [key, defaultParam(type, key, spec)])
        .filter(([, value]) => value !== undefined && value !== null && value !== ''));
}

function defaultParam(type, key, spec = {}) {
    const hidden = Number(AppState.flowchart?.model_info?.hidden_size ?? 0);
    if (key === 'intermediate_size') return 0;
    if (key === 'query_width_equals_hidden') return true;
    if (key === 'query_heads' && hidden > 0) {
        for (let heads = Math.min(64, hidden); heads >= 1; heads -= 1) {
            if (hidden % heads === 0 && hidden / heads <= 256) return heads;
        }
        return 1;
    }
    if (key === 'head_dim' && hidden > 0) {
        return Math.max(1, hidden / defaultParam(type, 'query_heads', spec));
    }
    if (key === 'kv_heads' && hidden > 0) {
        return Math.max(1, Math.min(defaultParam(type, 'query_heads', spec), 8));
    }
    return spec?.default;
}

function uniqueGroupId(kind) {
    const prefix = kind === 'prefix' ? 'prefix_custom'
        : kind === 'suffix' ? 'suffix_custom' : 'pattern_custom';
    const ids = new Set((AppState.flowchart?.nodes ?? []).map((node) => node.id));
    let index = 1;
    while (ids.has(`${prefix}_${index}`)) index += 1;
    return `${prefix}_${index}`;
}

function normalizeStageOrder(flowchart) {
    const nodes = flowchart.nodes ?? [];
    const embedding = nodes.filter((node) => node.role === 'embedding' || node.slot === 'embedding');
    const prefixes = nodes.filter((node) => isGroup(node) && node.group_kind === 'prefix');
    const patterns = nodes.filter((node) => isGroup(node) && node.group_kind === 'pattern');
    const suffixes = nodes.filter((node) => isGroup(node) && node.group_kind === 'suffix');
    const outputs = nodes.filter((node) => !isGroup(node) && !(node.role === 'embedding' || node.slot === 'embedding'));
    prefixes.forEach((node, index) => { node.label = `前置层 ${index + 1}`; });
    patterns.forEach((node, index) => { node.label = `循环 Pattern ${index + 1}`; });
    suffixes.forEach((node, index) => { node.label = `尾部层 ${index + 1}`; });
    flowchart.nodes = [...embedding, ...prefixes, ...patterns, ...suffixes, ...outputs];
    flowchart.edges = flowchart.nodes.slice(0, -1).map((node, index) => ({
        from: node.id, to: flowchart.nodes[index + 1].id,
    }));
}

function stageKindLabel(kind) {
    return kind === 'prefix' ? '前置层' : kind === 'suffix' ? '尾部层' : '循环 Pattern';
}

function commitStructureChange(reason, selectedId = null) {
    normalizeStageOrder(AppState.flowchart);
    markArchitectureCustomized();
    renderFlowchart();
    if (selectedId) selectNode(selectedId);
    Bus.emit(EVENTS.FLOWCHART_CHANGED, { reason, nodeId: selectedId });
}

function markArchitectureCustomized() {
    const info = AppState.flowchart?.model_info;
    const metadata = info?.metadata;
    if (!metadata || metadata.mapping_quality !== 'official_config_aligned') return;
    metadata.source_mapping_quality = metadata.mapping_quality;
    metadata.mapping_quality = 'customized_from_official';
}

/** 折叠 / 展开 Layer 组 */
export function toggleGroup(nodeId) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit || !isGroup(hit.node)) return;
    hit.node.collapsed = !hit.node.collapsed;
    renderFlowchart();
}

/** 批量折叠 / 展开所有 Layer 组 */
export function setAllGroupsCollapsed(collapsed) {
    walkNodes(AppState.flowchart, (node) => {
        if (isGroup(node)) node.collapsed = collapsed;
    });
    renderFlowchart();
}

/** 递归清除所有节点（含组内 children）的拖拽微调偏移并重绘 */
export function resetNodeOffsets() {
    let changed = false;
    walkNodes(AppState.flowchart, (node) => {
        if (node._offset_y !== undefined || node._offset_x !== undefined) {
            delete node._offset_y;
            delete node._offset_x;
            changed = true;
        }
    });
    if (changed) renderFlowchart();
    return changed;
}

/* ---------------------------------------------------------------- 视口控制 */

function measureHost() {
    const rect = host.getBoundingClientRect();
    hostSize = { w: Math.max(320, rect.width), h: Math.max(240, rect.height) };
}

function applyView() {
    if (!svg) return;
    const vw = hostSize.w / view.scale;
    const vh = hostSize.h / view.scale;
    svg.setAttribute('viewBox', `${round(view.x)} ${round(view.y)} ${round(vw)} ${round(vh)}`);
    Bus.emit(EVENTS.VIEW_CHANGED, { scale: view.scale });
}

/** 以画布中心为锚点缩放 */
export function zoomBy(factor) {
    zoomAt(hostSize.w / 2, hostSize.h / 2, factor);
}

/** 以屏幕坐标 (px, py) 为锚点缩放，保证锚点下的世界坐标不动 */
function zoomAt(px, py, factor) {
    const next = clamp(view.scale * factor, ZOOM_MIN, ZOOM_MAX);
    if (next === view.scale) return;
    const wx = view.x + px / view.scale;
    const wy = view.y + py / view.scale;
    view.scale = next;
    view.x = wx - px / next;
    view.y = wy - py / next;
    applyView();
}

/** 适应画布：内容整体可见并居中 */
export function fitView() {
    const bounds = getBounds(currentLayout);
    measureHost();
    // Long architectures stay readable by opening at a usable scale from the
    // top; the canvas can be panned instead of shrinking 93-layer views into
    // illegible thumbnails.
    // 注意：超长主干下 naturalScale 极小，比例会被钳到 0.55 下限——这是有意行为，
    // 连续点击“适应画布”得到相同 55% 属正常，缩放标签仍会同步刷新。
    const naturalScale = Math.min(hostSize.w / bounds.width, hostSize.h / bounds.height);
    const scale = clamp(Math.max(0.55, naturalScale), ZOOM_MIN, 1.15);
    view.scale = scale;
    view.x = bounds.x + bounds.width / 2 - hostSize.w / (2 * scale);
    view.y = bounds.y - VIEW_PADDING / 2;
    applyView();
}

/** 当前缩放比例 */
export function getScale() {
    return view.scale;
}

/* ---------------------------------------------------------------- 交互绑定 */

function bindInteractions() {
    // 滚轮缩放（Ctrl / 无修饰键均缩放，符合图形工具直觉）
    host.addEventListener('wheel', (event) => {
        event.preventDefault();
        const rect = host.getBoundingClientRect();
        const factor = Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0016));
        zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
    }, { passive: false });

    let drag = null;

    host.addEventListener('pointerdown', (event) => {
        if (event.button === 2) return; // 右键交给 contextmenu
        // 操作条 / 组级按钮：记录待分发动作，跳过节点拖拽与画布平移
        const actionEl = closestBySelector(event.target, '[data-action]');
        // 普通节点卡（dataset.kind === 'node'）允许拖拽微调；组框与空白处仍为画布平移
        const hitEl = closestBySelector(event.target, '[data-node-id]');
        // 内联残差卡（合成 id）无数据实体，不参与拖拽微调，点击仍可编辑
        const draggableNode = !actionEl && hitEl && hitEl.dataset.kind === 'node' && !hitEl.dataset.residualConn
            ? hitEl : null;
        drag = {
            startX: event.clientX,
            startY: event.clientY,
            viewX: view.x,
            viewY: view.y,
            scale: view.scale,
            moved: false,
            mode: null,
            target: event.target,
            actionEl,
            nodeEl: draggableNode,
            nodeId: draggableNode ? draggableNode.dataset.nodeId : null,
        };
        host.setPointerCapture?.(event.pointerId);
    });

    host.addEventListener('pointermove', (event) => {
        if (!drag) return;
        if (drag.actionEl) {
            // 按在操作按钮上：既不拖节点也不平移画布；位移超阈值视为放弃点击
            if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) >= 4) drag.moved = true;
            return;
        }
        const dx = event.clientX - drag.startX;
        const dy = event.clientY - drag.startY;
        if (!drag.moved && Math.hypot(dx, dy) < 4) return;
        if (!drag.moved) {
            drag.moved = true;
            const base = drag.nodeEl ? currentLayout[drag.nodeId] : null;
            drag.mode = base ? 'node' : 'pan';
            if (drag.mode === 'node') {
                drag.baseX = base.x;
                drag.baseY = base.y;
                host.classList.add('is-node-dragging');
            }
        }
        if (drag.mode === 'node') {
            // 轻量跟随：仅平移该节点 <g>，提交时才重绘；仅垂直方向，保持主干语义
            place(drag.nodeEl, drag.baseX, drag.baseY + dy / drag.scale);
            return;
        }
        host.classList.add('is-panning');
        view.x = drag.viewX - dx / drag.scale;
        view.y = drag.viewY - dy / drag.scale;
        applyView();
    });

    host.addEventListener('pointerup', (event) => {
        const state = drag;
        drag = null;
        host.classList.remove('is-panning');
        host.classList.remove('is-node-dragging');
        host.releasePointerCapture?.(event.pointerId);
        if (!state) return;
        if (state.mode === 'node') {
            commitNodeOffset(state, event);
            return;
        }
        if (state.moved) return; // 拖拽结束不触发点击

        // 操作条 / 组级按钮优先分发，分发后 return（不触发节点选中）
        const actionEl = closestBySelector(state.target, '[data-action]');
        if (actionEl) {
            dispatchNodeAction(actionEl);
            return;
        }

        const nodeEl = closestBySelector(state.target, '[data-node-id]');
        selectNode(nodeEl ? nodeEl.dataset.nodeId : null);
    });

    host.addEventListener('pointercancel', () => {
        const state = drag;
        drag = null;
        host.classList.remove('is-panning');
        host.classList.remove('is-node-dragging');
        // 节点拖拽中被取消时 <g> 停留在半空位移处，补一次重绘回弹
        if (state?.mode === 'node') renderFlowchart();
    });

    // 双击 → 算子替换（操作条按钮自身已带替换/插入入口，避免二次触发）
    host.addEventListener('dblclick', (event) => {
        if (closestBySelector(event.target, '[data-action]')) return;
        const nodeEl = closestBySelector(event.target, '[data-node-id]');
        if (!nodeEl) return;
        event.preventDefault();
        requestReplace(nodeEl.dataset.nodeId);
    });

    // 右键 → 算子替换
    host.addEventListener('contextmenu', (event) => {
        if (closestBySelector(event.target, '[data-action]')) return;
        const nodeEl = closestBySelector(event.target, '[data-node-id]');
        if (!nodeEl) return;
        event.preventDefault();
        selectNode(nodeEl.dataset.nodeId);
        requestReplace(nodeEl.dataset.nodeId);
    });
}

/**
 * 画布操作按钮统一分发：置灰按钮直接忽略；owner 取最近的 [data-node-id]
 * （卡片操作条 → 该卡；组级 add-op → 组框）。
 */
function dispatchNodeAction(actionEl) {
    if (actionEl.classList.contains('is-disabled')) return;
    const ownerEl = closestBySelector(actionEl, '[data-node-id]');
    const nodeId = ownerEl?.dataset.nodeId ?? null;
    if (!nodeId) return;
    switch (actionEl.dataset.action) {
        case 'up':
        case 'down':
            moveLayerOperation(nodeId, actionEl.dataset.action);
            return;
        case 'replace':
            requestReplace(nodeId);
            return;
        case 'remove': {
            const hit = findNodeById(AppState.flowchart, nodeId);
            if (hit?.parent && (hit.parent.children ?? []).length <= 1) {
                Bus.emit(EVENTS.TOAST, { type: 'info', message: '每组至少保留一个算子' });
                return;
            }
            removeLayerOperation(nodeId);
            return;
        }
        case 'insert': {
            const hit = findNodeById(AppState.flowchart, nodeId);
            if (!hit?.parent) return;
            openInsertPicker({
                groupId: hit.parent.id,
                afterOperationId: hit.node.operation_id ?? hit.node.role,
            });
            return;
        }
        case 'add-op':
            openInsertPicker({ groupId: nodeId, afterOperationId: null });
            return;
        default:
    }
}

/**
 * 提交节点微调拖拽：把屏幕位移换算为世界坐标后累加到 node._offset_y。
 * currentLayout 中的基准坐标已含既有偏移，故增量直接用指针位移即可。
 * _offset_y 为 _ 前缀运行时字段，stripRuntimeFields 会在提交后端前剥离。
 */
function commitNodeOffset(state, event) {
    const dyWorld = (event.clientY - state.startY) / state.scale;
    const hit = state.nodeId ? findNodeById(AppState.flowchart, state.nodeId) : null;
    if (!hit || isGroup(hit.node) || Math.abs(dyWorld) < 1) {
        renderFlowchart(); // 未命中节点或位移可忽略：回弹到原位
        return;
    }
    const next = Math.round(((hit.node._offset_y ?? 0) + dyWorld) * 100) / 100;
    if (next === 0) delete hit.node._offset_y;
    else hit.node._offset_y = next;
    renderFlowchart();
}

/** 发出算子替换请求（由 operator-panel 弹窗响应） */
function requestReplace(nodeId) {
    const hit = findNodeById(AppState.flowchart, nodeId);
    if (!hit || isGroup(hit.node)) return; // Layer 组本身不参与算子替换
    selectNode(nodeId);
    Bus.emit(EVENTS.NODE_REPLACE, hit.node);
}

/* ---------------------------------------------------------------- 小工具 */

/** 复用已存在的节点 <g>（保留 CSS transform 过渡），否则新建 */
function obtainNodeEl(nodeId, layer) {
    let element = nodeEls.get(nodeId);
    if (!element) {
        element = el('g');
        nodeEls.set(nodeId, element);
    }
    if (element.parentNode !== layer) layer.appendChild(element);
    return element;
}

function place(element, x, y) {
    element.style.transform = `translate(${round(x)}px, ${round(y)}px)`;
    element.setAttribute('transform', `translate(${round(x)}, ${round(y)})`);
}

function el(tag, attrs = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
        node.setAttribute(key, String(value));
    }
    return node;
}

function text(content, attrs = {}) {
    const node = el('text', attrs);
    node.textContent = content;
    return node;
}

/** 从事件目标向上找匹配元素（SVG 元素的 closest 在部分浏览器上不可靠，手动上溯） */
function closestBySelector(target, selector) {
    let cur = target;
    while (cur && cur !== host) {
        if (cur.matches?.(selector)) return cur;
        cur = cur.parentNode;
    }
    return null;
}

/** webfont（Noto Sans SC 等）就绪前测量标签可能偏短，就绪后重测一次 */
let fontRelayoutRegistered = false;
function ensureFontRelayout() {
    if (fontRelayoutRegistered || !document.fonts?.ready) return;
    fontRelayoutRegistered = true; // 全生命周期只注册一次
    document.fonts.ready.then(() => {
        renderFlowchart(); // 全量重渲染重新测量，比单独重测截断标签更彻底
    }).catch(() => { /* 字体加载失败不影响已渲染内容 */ });
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function round(value) {
    return Math.round(value * 100) / 100;
}
