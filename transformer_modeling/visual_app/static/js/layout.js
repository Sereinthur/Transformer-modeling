/**
 * layout.js —— 线性主干布局参数与画布工具
 *
 * 渲染几何统一收敛到 LINEAR_LAYOUT_DEFAULTS：flowchart.js 不再持有任何
 * 布局字面量，后续布局面板通过 applyLinearLayoutOverrides 调整参数。
 */

/**
 * 线性主干渲染默认几何参数（单位：SVG 用户坐标 = px）。
 * 数值固化自当前渲染字面量，默认渲染结果与收敛前完全一致。
 */
export const LINEAR_LAYOUT_DEFAULTS = Object.freeze({
    canvasW: 900,      // 画布宽（viewBox 宽度）
    cardW: 220,        // 算子卡片宽
    cardH: 42,         // 算子卡片高
    cardGap: 18,       // 卡片之间的垂直间距
    groupGap: 44,      // 阶段（Layer 组）之间的附加间距
    topY: 66,          // 首张卡片的起始 Y
    frameX: 292,       // 阶段框左缘（默认画布下等价于 centerX - frameW / 2）
    frameW: 316,       // 阶段框宽度
});

/** 各参数的合法区间（per-key min/max 钳制） */
const CLAMP_RULES = {
    canvasW: [600, 1600],
    cardW: [180, 320],
    cardH: [32, 80],
    cardGap: [8, 60],
    groupGap: [20, 120],
    topY: [16, 200],
    frameX: [0, 1200],
    frameW: [240, 480],
};

/** 画布边缘留白（getBounds / fitView 共用，数值固化自旧 LAYOUT.PADDING） */
export const VIEW_PADDING = 60;

/** 当前生效的布局覆盖参数（模块级状态；持久化由 app.js 的 vm-linear-layout 存取） */
let overrides = {};

/**
 * 合并布局覆盖参数。仅接受 LINEAR_LAYOUT_DEFAULTS 已有的键与有限数值。
 * @param {object} [partial] 形如 { cardGap: 24 } 的部分覆盖
 * @returns {object} 合并并钳制后的当前生效布局
 */
export function applyLinearLayoutOverrides(partial) {
    for (const [key, value] of Object.entries(partial ?? {})) {
        if (!(key in LINEAR_LAYOUT_DEFAULTS)) continue;
        if (typeof value !== 'number' || !Number.isFinite(value)) continue;
        overrides[key] = value;
    }
    return getLinearLayout();
}

/** 清空覆盖参数，恢复默认布局。 */
export function resetLinearLayout() {
    overrides = {};
    return getLinearLayout();
}

/**
 * 当前生效的布局参数：默认值与 overrides 合并后按 per-key 区间钳制。
 * 附加推导量：centerX = canvasW / 2；frameX 保持阶段框水平居中。
 * @returns {object} 只含数值键的参数对象
 */
export function getLinearLayout() {
    const layout = {};
    for (const [key, fallback] of Object.entries(LINEAR_LAYOUT_DEFAULTS)) {
        const raw = key in overrides ? overrides[key] : fallback;
        const [min, max] = CLAMP_RULES[key] ?? [-Infinity, Infinity];
        layout[key] = Math.min(max, Math.max(min, raw));
    }
    layout.centerX = layout.canvasW / 2;
    // 阶段框在现状中即以画布中心对称（292 + 316 / 2 = 450），保持该语义
    layout.frameX = layout.centerX - layout.frameW / 2;
    return layout;
}

/** 是否为 Layer 组（带子节点的容器节点） */
export function isGroup(node) {
    return node?.type === 'layer_group' || Array.isArray(node?.children);
}

/**
 * 计算内容包围盒（含边缘留白），供画布尺寸与「适应画布」使用。
 * @returns {{x:number,y:number,width:number,height:number}}
 */
export function getBounds(layout) {
    const boxes = Object.values(layout);
    if (!boxes.length) {
        return { x: 0, y: 0, width: 620, height: 400 };
    }
    let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
    for (const box of boxes) {
        minX = Math.min(minX, box.x);
        minY = Math.min(minY, box.y);
        maxX = Math.max(maxX, box.x + box.width);
        maxY = Math.max(maxY, box.y + box.height);
    }
    const pad = VIEW_PADDING;
    return {
        x: minX - pad,
        y: minY - pad,
        width: (maxX - minX) + pad * 2,
        height: (maxY - minY) + pad * 2,
    };
}
