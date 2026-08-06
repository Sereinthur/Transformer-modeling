/**
 * api.js —— 全局状态管理、后端 API 封装、轻量事件总线与单位常量
 *
 * 约定：AppState 是整个应用的唯一数据源；各模块通过 Bus 通信，避免相互直接依赖。
 */

/* ---------------------------------------------------------------- 全局状态 */

export const AppState = {
    flowchart: null,        // Schema v3 线性算子视图 {nodes, edges, model_info}
    sidebarConfig: {},      // 左侧栏产出的 {hardware, serving, execution, parallelism}
    selectedNode: null,     // 当前选中节点对象（流程图节点引用）
    operatorSchemas: null,  // 算子 schema 缓存（按算子类型索引，已从 operators 字段解包）
    slotOperators: null,    // 算子类别索引（仅用于颜色与筛选，不限制层内顺序）
    dtypeChoices: null,     // 可选精度列表
    operatorCatalog: null,  // 算子目录缓存 {schema_version, operators:[...]}
    results: null,          // 最新估算结果
    baseConfig: null,       // 当前完整 config（作为 flowchart_to_config 的底座）
    presets: [],            // 模型预设列表
    currentPresetId: null,  // 当前预设 id
    presetWarnings: [],     // 预设解析告警（未建模特性等）
};

/* ---------------------------------------------------------------- 单位常量
 * 后端 config 一律使用原始单位（bytes / bytes per second / ops per second / seconds），
 * 界面上给用户展示的是友好单位。这里集中定义换算系数，避免各处硬编码。
 *   容量类按 1024³（与 85899345920 bytes = 80 GB 的口径一致）
 *   算力/带宽按十进制（100 TOPS = 1e14 ops/s，2 TB/s = 2e12 bytes/s）
 */
export const UNITS = {
    GB: 1024 ** 3,   // 容量：GB → bytes
    TOPS: 1e12,      // 算力：TOPS → ops/s
    TBPS: 1e12,      // 带宽：TB/s → bytes/s
    GBPS: 1e9,       // 互联带宽：GB/s → bytes/s
    US: 1e-6,        // 延迟：μs → s
    MS: 1e-3,        // 延迟：ms → s
};

/* ---------------------------------------------------------------- 事件总线 */

const listeners = new Map();

export const Bus = {
    /** 订阅事件，返回取消订阅函数 */
    on(event, handler) {
        if (!listeners.has(event)) listeners.set(event, new Set());
        listeners.get(event).add(handler);
        return () => listeners.get(event).delete(handler);
    },
    /** 触发事件（单个 handler 抛错不影响其它 handler） */
    emit(event, payload) {
        const set = listeners.get(event);
        if (!set) return;
        for (const handler of [...set]) {
            try {
                handler(payload);
            } catch (err) {
                console.error(`[Bus] 事件 ${event} 的处理器出错：`, err);
            }
        }
    },
};

/** 应用内事件名集中登记，防止拼写漂移 */
export const EVENTS = {
    NODE_SELECTED: 'node:selected',       // 节点被选中 → 右侧面板刷新
    NODE_REPLACE: 'node:replace',         // 请求替换算子 → 弹出算子选择框
    FLOWCHART_CHANGED: 'flowchart:changed', // 流程图结构/参数变化 → 重绘、结果失效
    CONFIG_CHANGED: 'config:changed',     // 左侧栏配置变化
    VIEW_CHANGED: 'view:changed',         // 画布缩放/平移变化
    RESULTS_READY: 'results:ready',       // 估算完成
    TOAST: 'ui:toast',                    // 通知
};

/* ---------------------------------------------------------------- HTTP 封装 */

/** 后端返回非 2xx 时携带的错误信息 */
export class ApiError extends Error {
    constructor(message, status, detail) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.detail = detail;
    }
}

/**
 * 统一请求入口：负责 JSON 序列化、错误解析与超时。
 * @param {string} path  接口路径
 * @param {object} [options] {method, body, timeout}
 */
async function request(path, options = {}) {
    const { method = 'GET', body, timeout = 60000 } = options;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    let response;
    try {
        response = await fetch(path, {
            method,
            headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
            body: body !== undefined ? JSON.stringify(body) : undefined,
            signal: controller.signal,
        });
    } catch (err) {
        clearTimeout(timer);
        if (err.name === 'AbortError') {
            throw new ApiError(`请求超时：${path}`, 0, null);
        }
        throw new ApiError(`网络请求失败：${path}（后端服务是否已启动？）`, 0, String(err));
    }
    clearTimeout(timer);

    const text = await response.text();
    let payload = null;
    if (text) {
        try {
            payload = JSON.parse(text);
        } catch {
            payload = text;
        }
    }

    if (!response.ok) {
        const detail = payload && typeof payload === 'object'
            ? (payload.detail ?? payload.error ?? payload.message ?? payload)
            : payload;
        throw new ApiError(
            `${path} 返回 ${response.status}：${formatDetail(detail)}`,
            response.status,
            detail,
        );
    }
    return payload;
}

/** 把后端返回的 detail（可能是对象/数组）压成一行可读文本 */
function formatDetail(detail) {
    if (detail == null) return '无详细信息';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map(formatDetail).join('；');
    if (typeof detail === 'object') {
        if (detail.msg) return `${(detail.loc || []).join('.')} ${detail.msg}`.trim();
        try {
            return JSON.stringify(detail);
        } catch {
            return String(detail);
        }
    }
    return String(detail);
}

/* ---------------------------------------------------------------- 接口方法 */

/** 算子目录：{schema_version, operators:[{type, chinese_name, slot, implementations}]} */
export async function fetchOperatorCatalog() {
    const data = await request('/api/operator-catalog');
    AppState.operatorCatalog = data;
    return data;
}

/**
 * 算子 schema。后端返回 {schema_version, dtype_choices, slot_operators, operators}，
 * 其中 operators 才是 {算子类型: {slot, chinese_name, implementations, params}}。
 */
export async function fetchOperatorSchemas() {
    const data = await request('/api/operator-schemas');
    const operators = data?.operators ?? data ?? {};
    AppState.operatorSchemas = operators;
    AppState.slotOperators = data?.slot_operators ?? null;
    AppState.dtypeChoices = data?.dtype_choices ?? null;
    return operators;
}

/** 模型预设列表 */
export async function fetchModelPresets() {
    const data = await request('/api/model-presets');
    AppState.presets = data?.presets ?? [];
    return AppState.presets;
}

/**
 * 预设 → 完整 config。
 * 后端返回 {resolved_model, default_max_sequence_length, warnings, config}，config 才是可直接提交的整份配置。
 */
export async function resolvePreset(presetId) {
    const data = await request('/api/resolve-preset', {
        method: 'POST',
        body: { preset_id: presetId },
    });
    AppState.presetWarnings = Array.isArray(data?.warnings) ? data.warnings : [];
    return data?.config ?? data;
}

/** config → 流程图 */
export async function configToFlowchart(config) {
    return request('/api/config-to-flowchart', { method: 'POST', body: config });
}

/**
 * 流程图 + 左侧栏配置 → 完整 config。
 * 后端把完整 config 平铺在顶层，同时附一份 config 别名，这里统一取别名。
 */
export async function flowchartToConfig(flowchart, sidebarConfig) {
    const data = await request('/api/flowchart-to-config', {
        method: 'POST',
        body: {
            flowchart: stripRuntimeFields(flowchart),
            config: sidebarConfig ?? {},
        },
    });
    return data?.config ?? data;
}

/** config → 估算结果 */
export async function runEstimate(config) {
    return request('/api/estimate', { method: 'POST', body: config });
}

/* ---------------------------------------------------------------- 工具函数 */

/**
 * 去掉前端加在节点上的运行时字段（折叠状态等），避免污染后端请求体。
 */
export function stripRuntimeFields(flowchart) {
    if (!flowchart) return flowchart;
    const cleanNode = (node) => {
        const copy = {};
        for (const [key, value] of Object.entries(node)) {
            if (key === 'collapsed' || key === 'details_expanded' || key.startsWith('_')) continue;
            if (key === 'children' && Array.isArray(value)) copy[key] = value.map(cleanNode);
            else copy[key] = value;
        }
        return copy;
    };
    // 连线与覆盖层号都是视图派生数据。后端只接受有序算子节点，避免把
    // 任意前端图边误当成可估算的计算拓扑。
    const { connection_view, architecture_view, edges, ...persistable } = flowchart;
    const cleaned = {
        ...persistable,
        nodes: (flowchart.nodes ?? []).map(cleanNode),
    };
    return cleaned;
}

/** 深拷贝（结构化克隆优先，退化到 JSON） */
export function deepClone(value) {
    if (value == null) return value;
    if (typeof structuredClone === 'function') {
        try {
            return structuredClone(value);
        } catch { /* 含不可克隆对象时退化 */ }
    }
    return JSON.parse(JSON.stringify(value));
}

/** 按 'a.b.c' 路径读取嵌套值 */
export function getPath(obj, path) {
    if (!obj || !path) return undefined;
    return path.split('.').reduce((cur, key) => (cur == null ? undefined : cur[key]), obj);
}

/** 按 'a.b.c' 路径写入嵌套值（沿途自动补对象） */
export function setPath(obj, path, value) {
    const keys = path.split('.');
    let cur = obj;
    for (let i = 0; i < keys.length - 1; i += 1) {
        const key = keys[i];
        if (cur[key] == null || typeof cur[key] !== 'object') cur[key] = {};
        cur = cur[key];
    }
    cur[keys[keys.length - 1]] = value;
    return obj;
}

/** 遍历流程图所有节点（含 layer_group 子节点），回调 (node, parent) */
export function walkNodes(flowchart, visit) {
    const walk = (nodes, parent) => {
        for (const node of nodes ?? []) {
            visit(node, parent);
            if (Array.isArray(node.children)) walk(node.children, node);
        }
    };
    walk(flowchart?.nodes, null);
}

/** 按 id 查找节点，返回 {node, parent} */
export function findNodeById(flowchart, nodeId) {
    let found = null;
    walkNodes(flowchart, (node, parent) => {
        if (!found && node.id === nodeId) found = { node, parent };
    });
    return found;
}
