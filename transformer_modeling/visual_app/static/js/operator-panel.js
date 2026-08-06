/**
 * operator-panel.js —— 右侧算子参数面板 + 算子替换弹窗
 *
 * 面板内容由「选中节点 + 算子 schema」共同决定：
 *   schema.params = {参数名: {type, min, max, default, label, required, options}}
 */

import { AppState, Bus, EVENTS, findNodeById } from './api.js';
import {
    categoryOf, insertLayerOperation, moveLayerOperation, removeLayerOperation,
    setLayerGroupKind, setPatternCycleCount, updateNodeField, updateNodeParams, updateNodeType, selectNode,
} from './flowchart.js';

let bodyEl = null;
let tagEl = null;
let fieldControlCounter = 0;

/** 初始化面板，订阅节点选中 / 算子替换事件 */
export function initOperatorPanel(container, tagContainer) {
    bodyEl = container;
    tagEl = tagContainer ?? null;
    Bus.on(EVENTS.NODE_SELECTED, (node) => renderOperatorPanel(node));
    Bus.on(EVENTS.NODE_REPLACE, (node) => openReplaceModal(node));
    renderOperatorPanel(null);
}

/* ---------------------------------------------------------------- 面板渲染 */

/**
 * 根据选中节点渲染参数表单。
 * @param {object|null} node 选中节点
 * @param {object} [schemas] 算子 schema 集合
 */
export function renderOperatorPanel(node, schemas = AppState.operatorSchemas) {
    if (!bodyEl) return;
    bodyEl.innerHTML = '';
    if (tagEl) tagEl.textContent = '';

    if (!node) {
        bodyEl.appendChild(buildEmptyState());
        return;
    }

    const schema = schemas?.[node.type] ?? null;
    const category = categoryOf(node);
    if (tagEl) tagEl.textContent = node.slot ? `类别 · ${node.slot}` : '';

    bodyEl.appendChild(buildHeader(node, schema, category));

    if (node.type === 'token_embedding') {
        bodyEl.appendChild(buildEmbeddingShapeSection(node));
    }

    if (node.type === 'layer_group' || Array.isArray(node.children)) {
        bodyEl.appendChild(buildGroupSection(node));
        return;
    }

    // 长模型的循环层段不一定同时落在当前可视区。把所属层组的重复次数
    // 放到任一子算子的参数面板中，用户选中 KDA / MLA / HCA / CSA 后就能
    // 直接修改 ×N，无需先平移到很长的层组边框上。
    const parent = findNodeById(AppState.flowchart, node.id)?.parent;
    if (parent?.type === 'layer_group' || Array.isArray(parent?.children)) {
        bodyEl.appendChild(buildParentGroupSection(parent));
    }

    // implementation 下拉
    const impls = schema?.implementations ?? [];
    // "default" alone is not a user-selectable implementation.  Showing it
    // made parameterised operators such as Embedding look like empty cards.
    if (impls.length > 1) {
        const section = sectionEl('实现方式');
        section.appendChild(buildSelectRow({
            label: '算子实现',
            value: node.implementation ?? impls[0],
            options: impls,
            hint: '影响算子的建模公式与效率取值',
            onChange: (value) => updateNodeField(node.id, 'implementation', value),
        }));
        bodyEl.appendChild(section);
    }

    // 参数表单
    const params = schema?.params ?? {};
    const entries = Object.entries(params);
    const section = sectionEl(`算子参数${entries.length ? `（${entries.length}）` : ''}`);
    if (!entries.length) {
        section.appendChild(hintEl(schema
            ? (schema.description ?? '该算子无可调参数，形状由上下文推导。')
            : `未在 schema 中找到算子类型 ${node.type}，仅展示已有参数。`));
        // schema 缺失时，退化为按现有 params 生成数字/文本输入
        for (const [key, value] of Object.entries(node.params ?? {})) {
            section.appendChild(buildParamRow(node, key, inferSpec(value), value));
        }
    } else {
        const groups = [
            ['performance', '性能估算'],
            ['inheritance', '性能覆盖与继承'],
            ['numerical', '数值语义（当前不影响性能）'],
        ];
        for (const [effect, title] of groups) {
            const selected = entries.filter(([, spec]) => {
                const category = spec?.effect === 'numerical' ? 'numerical'
                    : spec?.inherit_from ? 'inheritance' : 'performance';
                return category === effect;
            });
            if (!selected.length) continue;
            const heading = document.createElement('p');
            heading.className = 'param-effect-heading';
            heading.textContent = title;
            section.appendChild(heading);
            for (const [key, spec] of selected) {
                const value = node.params?.[key] ?? spec?.default;
                section.appendChild(buildParamRow(node, key, spec ?? {}, value));
            }
        }
    }
    bodyEl.appendChild(section);

    // 节点标识
    const meta = document.createElement('div');
    meta.className = 'node-meta';
    meta.append(
        metaRow('节点 ID', node.id),
        metaRow('算子类型', node.type ?? '—'),
        metaRow('算子类别', node.slot ?? '—'),
    );
    bodyEl.appendChild(meta);
}

function buildEmptyState() {
    const wrap = document.createElement('div');
    wrap.className = 'empty-state';
    const glyph = document.createElement('div');
    glyph.className = 'empty-glyph';
    glyph.textContent = '◇';
    const p1 = document.createElement('p');
    p1.textContent = '在流程图中点击任意算子节点，即可在此编辑其参数。';
    const p2 = document.createElement('p');
    p2.className = 'hint';
    p2.textContent = '双击或右键节点可直接更换算子；滚轮缩放，拖拽平移。';
    wrap.append(glyph, p1, p2);
    return wrap;
}

/**
 * Embedding 的形状来自模型维度；算子卡可以覆盖表行数和权重精度，但
 * 在当前 decoder-only 线性主干中，输出宽度必须接到同一个 hidden state。
 */
function buildEmbeddingShapeSection(node) {
    const model = AppState.flowchart?.model_info ?? {};
    const inheritedRows = Number(model.padded_vocab_size ?? model.vocab_size ?? 0);
    const hidden = Number(model.hidden_size ?? 0);
    const configuredRows = Number(node.params?.vocab_size ?? 0);
    const configuredWidth = Number(node.params?.embedding_dim ?? 0);
    const rows = configuredRows > 0 ? configuredRows : inheritedRows;
    const width = configuredWidth > 0 ? configuredWidth : hidden;
    const tied = node.params?.tied_lm_head !== false;
    const parameters = rows > 0 && width > 0 ? rows * width : 0;
    const section = sectionEl('派生形状');
    section.append(
        metaRow('Embedding 表', rows && width ? `${rows} × ${width}` : '等待模型维度'),
        metaRow('参数量', parameters ? formatCompactCount(parameters) : '—'),
        metaRow('LM Head', tied ? '共享同一张表' : '独立权重'),
        hintEl('词表行数和权重精度可在下方覆盖；输出宽度必须与主干 hidden size 对齐。'),
    );
    return section;
}

function formatCompactCount(value) {
    if (value >= 1e9) return `${(value / 1e9).toFixed(3)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(3)}M`;
    return String(value);
}

/**
 * 子算子视图中的层组快捷设置。它只编辑父 LayerGroup 的 repeat，
 * 不改变当前选中的独立算子，也不会把层序折叠成固定槽位。
 */
function buildParentGroupSection(group) {
    const section = sectionEl('所属层组');
    const stats = group._structure ?? {};
    const kind = group.group_kind === 'prefix' ? '前置段'
        : group.group_kind === 'suffix' ? '尾部段' : '循环 Pattern';
    section.appendChild(metaRow('层段', kind));
    if (stats.first_layer) {
        const last = stats.last_layer && stats.last_layer !== stats.first_layer
            ? `–${stats.last_layer}` : '';
        section.appendChild(metaRow('实际覆盖', `L${stats.first_layer}${last}`));
    }
    section.appendChild(buildNumberRow({
        label: '层内重复数 ×N',
        value: Number(group.repeat ?? 1),
        min: 1,
        step: 1,
        hint: group.group_kind === 'pattern'
            ? '该 Pattern 段在每个循环单元内连续占用的层数。'
            : '该层段连续包含的层数。',
        onChange: (value) => updateNodeField(
            group.id, 'repeat', Math.max(1, Math.round(value)),
        ),
    }));
    if (group.group_kind === 'pattern') {
        section.appendChild(buildPatternCycleRow(group));
        section.appendChild(patternRelationHint(group));
    }
    const selectGroup = document.createElement('button');
    selectGroup.type = 'button';
    selectGroup.className = 'btn btn-ghost btn-mini';
    selectGroup.textContent = '编辑整个层组';
    selectGroup.addEventListener('click', () => selectNode(group.id));
    section.appendChild(selectGroup);
    return section;
}

/** 循环重复数是整个 Pattern 单元的展开次数，不等同于该组内部的 repeat。 */
function buildPatternCycleRow(group) {
    const stats = group._structure ?? {};
    const cycles = Math.max(1, Number(stats.full_cycle_count ?? 1));
    const cycleLength = Math.max(1, Number(AppState.flowchart?.model_info?.pattern_cycle_length ?? 1));
    return buildNumberRow({
        label: '循环重复数 ×C',
        value: cycles,
        min: 1,
        step: 1,
        hint: `控制整个 Pattern 单元的重复轮数（当前每轮 ${cycleLength} 层）；会同步更新模型总层数与 Pattern 层号范围。`,
        onChange: (value) => setPatternCycleCount(Math.max(1, Math.round(value))),
    });
}

/** 一条 Pattern 段的真实展开关系，而非整个循环区的总层数关系。 */
function patternRelationHint(group) {
    const stats = group._structure ?? {};
    const inner = Math.max(1, Number(group.repeat ?? 1));
    const cycles = Math.max(1, Number(stats.full_cycle_count ?? 1));
    const partial = Math.max(0, Number(stats.partial_occurrences ?? 0));
    const complete = inner * cycles;
    return hintEl(`关系：该段每轮层数 ${inner} × 完整循环次数 ${cycles} = ${complete} 层`
        + (partial ? `，再加末轮截断落入本段的 ${partial} 层。` : '。'));
}

/** 面板头部：中文名 + 类型代码 + 更换算子按钮 */
function buildHeader(node, schema, category) {
    const head = document.createElement('div');
    head.className = `node-head cat-${category}`;

    const title = document.createElement('div');
    title.className = 'node-title';
    const name = document.createElement('h3');
    name.textContent = schema?.chinese_name ?? node.label ?? node.type ?? node.id;
    const code = document.createElement('code');
    code.textContent = node.type ?? '—';
    title.append(name, code);
    head.appendChild(title);

    if (node.type !== 'layer_group' && !Array.isArray(node.children)) {
        const swap = document.createElement('button');
        swap.className = 'btn btn-ghost btn-swap';
        swap.type = 'button';
        swap.textContent = '⇄ 更换算子';
        swap.title = '替换为另一个已注册算子；参数按新算子契约重新校验';
        swap.addEventListener('click', () => openReplaceModal(node));
        head.appendChild(swap);
    }
    return head;
}

/** Layer 组：重复次数 + 子算子快捷列表 */
function buildGroupSection(node) {
    const stats = node._structure ?? {};
    const summary = sectionEl('结构统计');
    const kind = node.group_kind === 'prefix' ? '前置层段（按顺序展开）'
        : node.group_kind === 'suffix' ? '尾部层段（锚定模型末尾）'
            : '循环 Pattern（填充中间层）';
    summary.appendChild(metaRow('阶段语义', kind));
    if (stats.kind === 'prefix') {
        summary.appendChild(metaRow('覆盖层号', `L${stats.first_layer}–L${stats.last_layer}`));
        summary.appendChild(metaRow('实际层数', String(stats.occurrences ?? node.repeat ?? 1)));
    } else if (stats.kind === 'pattern') {
        summary.appendChild(metaRow('循环内位置', `第 ${stats.cycle_position_start}–${stats.cycle_position_end} 层`));
        summary.appendChild(metaRow('该 Pattern 段实际层数', `${stats.occurrences ?? 0} 层`));
        const tail = stats.partial_occurrences ? `（尾部截断 ${stats.partial_occurrences} 层）` : '';
        summary.appendChild(metaRow('层号范围', `L${stats.first_layer}–L${stats.last_layer || '—'} ${tail}`));
    } else if (stats.kind === 'suffix') {
        summary.appendChild(metaRow('覆盖层号', `L${stats.first_layer}–L${stats.last_layer}`));
        summary.appendChild(metaRow('实际层数', String(stats.occurrences ?? node.repeat ?? 1)));
    }

    const quality = AppState.flowchart?.model_info?.metadata?.mapping_quality;
    if (quality) summary.appendChild(metaRow('映射状态', quality));

    const section = sectionEl('层组设置');

    section.appendChild(buildSelectRow({
        label: '阶段类型',
        value: node.group_kind ?? 'pattern',
        options: [
            { value: 'prefix', label: '前置层（先按顺序展开）' },
            { value: 'pattern', label: '循环 Pattern（填充中间层）' },
            { value: 'suffix', label: '尾部层（锚定模型末尾）' },
        ],
        hint: '固定顺序为前置段、循环段、尾部段；至少保留一个循环 Pattern。',
        onChange: (value) => setLayerGroupKind(node.id, value),
    }));

    section.appendChild(buildNumberRow({
        label: '层内重复数 ×N',
        value: Number(node.repeat ?? 1),
        min: 1,
        step: 1,
        hint: '该 Pattern 段在每个循环单元内连续占用的层数。',
        onChange: (value) => updateNodeField(node.id, 'repeat', Math.max(1, Math.round(value))),
    }));
    if (node.group_kind === 'pattern') {
        section.appendChild(buildPatternCycleRow(node));
        section.appendChild(patternRelationHint(node));
    }

    const children = node.children ?? [];
    const backboneControls = document.createElement('div');
    backboneControls.className = 'backbone-controls';
    const insertOp = document.createElement('button');
    insertOp.type = 'button';
    insertOp.className = 'btn btn-ghost';
    insertOp.textContent = '+ 添加算子';
    insertOp.title = '在本组末尾插入一个算子（从候选列表中选择类型）';
    insertOp.addEventListener('click', () => openInsertPicker({ groupId: node.id, afterOperationId: null }));
    backboneControls.appendChild(insertOp);
    const list = document.createElement('div');
    list.className = 'child-list';
    for (const child of children) {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `child-item cat-${categoryOf(child)}`;
        const label = document.createElement('span');
        label.className = 'child-name';
        label.textContent = child.label ?? child.type;
        const code = document.createElement('code');
        code.textContent = child.type;
        item.append(label, code);
        item.addEventListener('click', () => selectNode(child.id));
        const row = document.createElement('div');
        row.className = 'backbone-op-row';
        const tools = document.createElement('span');
        tools.className = 'backbone-op-tools';
        for (const [direction, glyph] of [['up', '↑'], ['down', '↓']]) {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'mini-btn'; button.textContent = glyph;
            button.title = `Move ${direction}`;
            button.addEventListener('click', (event) => { event.stopPropagation(); moveLayerOperation(child.id, direction); });
            tools.appendChild(button);
        }
        const add = document.createElement('button');
        add.type = 'button'; add.className = 'mini-btn'; add.textContent = '+';
        add.title = '在此算子之后插入（从候选列表中选择类型）';
        add.addEventListener('click', (event) => {
            event.stopPropagation();
            openInsertPicker({ groupId: node.id, afterOperationId: child.operation_id ?? child.role });
        });
        const remove = document.createElement('button');
        remove.type = 'button'; remove.className = 'mini-btn is-danger'; remove.textContent = '×'; remove.title = 'Delete operator';
        remove.addEventListener('click', (event) => { event.stopPropagation(); removeLayerOperation(child.id); });
        tools.append(add, remove);
        row.append(item, tools);
        list.appendChild(row);
    }
    const listWrap = sectionEl(`层内算子（${children.length}）`);
    listWrap.append(backboneControls, list);

    const frag = document.createDocumentFragment();
    frag.append(summary, section, listWrap);
    return frag;
}

/* ---------------------------------------------------------------- 表单行 */

/** 依据 schema 参数类型分派控件 */
function buildParamRow(node, key, spec, value) {
    const type = String(spec?.type ?? 'float').toLowerCase();
    const label = spec?.label ?? key;
    const required = spec?.required === true;
    const hint = buildHint(spec);

    // 后端枚举型 spec 用 type="enum" + choices；兼容 select/options 写法。
    const choices = spec?.choices ?? spec?.options;
    if (type === 'enum' || type === 'select' || Array.isArray(choices)) {
        return buildSelectRow({
            label, required, hint,
            value: value ?? spec?.default ?? '',
            options: (choices ?? []).map(optionOf),
            code: key,
            onChange: (next) => updateNodeParams(node.id, { [key]: next }),
        });
    }
    if (type === 'bool' || type === 'boolean') {
        return buildToggleRow({
            label, required, hint, code: key,
            value: value === true,
            onChange: (next) => updateNodeParams(node.id, { [key]: next }),
        });
    }
    if (type === 'str' || type === 'string' || type === 'text') {
        return buildTextRow({
            label, required, hint, code: key,
            value: value ?? '',
            onChange: (next) => updateNodeParams(node.id, { [key]: next }),
        });
    }
    const isInt = type === 'int' || type === 'integer';
    return buildNumberRow({
        label, required, hint, code: key,
        value: value ?? spec?.default ?? 0,
        min: spec?.min,
        max: spec?.max,
        step: spec?.step ?? (isInt ? 1 : 0.01),
        onChange: (next) => {
            const clamped = clampToSpec(isInt ? Math.round(next) : next, spec);
            updateNodeParams(node.id, { [key]: clamped });
        },
    });
}

function buildRowShell({ label, required, hint, code }) {
    const row = document.createElement('div');
    row.className = 'form-row';
    if (code) row.dataset.field = code;
    const head = document.createElement('div');
    head.className = 'form-label';
    const text = document.createElement('span');
    text.textContent = label;
    head.appendChild(text);
    if (required) {
        const star = document.createElement('em');
        star.className = 'req';
        star.textContent = '必填';
        head.appendChild(star);
    }
    if (code) {
        const codeEl = document.createElement('code');
        codeEl.className = 'form-code';
        codeEl.textContent = code;
        head.appendChild(codeEl);
    }
    row.appendChild(head);
    const control = document.createElement('div');
    control.className = 'form-control';
    row.appendChild(control);
    if (hint) row.appendChild(hintEl(hint));
    return { row, control };
}

function labelControl(control, options) {
    fieldControlCounter += 1;
    control.id = `vm-field-${fieldControlCounter}`;
    control.setAttribute('aria-label', options.label ?? options.code ?? '参数');
}

export function buildNumberRow(options) {
    const { row, control } = buildRowShell(options);
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'ctl ctl-input';
    labelControl(input, options);
    input.value = options.value ?? '';
    if (options.min != null) input.min = options.min;
    if (options.max != null) input.max = options.max;
    if (options.step != null) input.step = options.step;
    let lastCommitted = Number(options.value);
    const commit = () => {
        const parsed = Number(input.value);
        if (!Number.isFinite(parsed)) {
            input.value = options.value ?? '';
            return;
        }
        if (Object.is(parsed, lastCommitted)) return;
        lastCommitted = parsed;
        options.onChange(parsed);
    };
    // change 覆盖常规数值控件；blur 保证用户改完后直接点击画布、
    // 操作条或层组按钮时也会提交，而不会只留下未写回的输入框文本。
    input.addEventListener('change', commit);
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        commit();
        input.blur();
    });
    control.appendChild(input);
    if (options.unit) {
        const unit = document.createElement('span');
        unit.className = 'unit';
        unit.textContent = options.unit;
        control.appendChild(unit);
    }
    return row;
}

export function buildTextRow(options) {
    const { row, control } = buildRowShell(options);
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'ctl ctl-input';
    labelControl(input, options);
    input.value = options.value ?? '';
    let lastCommitted = String(options.value ?? '');
    const commit = () => {
        if (input.value === lastCommitted) return;
        lastCommitted = input.value;
        options.onChange(input.value);
    };
    input.addEventListener('change', commit);
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        commit();
        input.blur();
    });
    control.appendChild(input);
    return row;
}

export function buildSelectRow(options) {
    const { row, control } = buildRowShell(options);
    const select = document.createElement('select');
    select.className = 'ctl ctl-select';
    labelControl(select, options);
    for (const raw of options.options ?? []) {
        const value = typeof raw === 'object' ? (raw.value ?? raw.id) : raw;
        const label = typeof raw === 'object' ? (raw.label ?? raw.name ?? value) : raw;
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = String(label);
        select.appendChild(option);
    }
    select.value = String(options.value ?? '');
    if (select.selectedIndex < 0 && select.options.length) select.selectedIndex = 0;
    select.addEventListener('change', () => options.onChange(select.value));
    control.appendChild(select);
    return row;
}

export function buildToggleRow(options) {
    const { row, control } = buildRowShell(options);
    row.classList.add('is-toggle');
    const label = document.createElement('label');
    label.className = 'switch';
    const input = document.createElement('input');
    input.type = 'checkbox';
    labelControl(input, options);
    input.checked = options.value === true;
    const track = document.createElement('span');
    track.className = 'switch-track';
    const state = document.createElement('span');
    state.className = 'switch-state';
    state.textContent = input.checked ? '开启' : '关闭';
    input.addEventListener('change', () => {
        state.textContent = input.checked ? '开启' : '关闭';
        options.onChange(input.checked);
    });
    label.append(input, track, state);
    control.appendChild(label);
    return row;
}

export function buildRangeRow(options) {
    const { row, control } = buildRowShell(options);
    const input = document.createElement('input');
    input.type = 'range';
    input.className = 'ctl ctl-range';
    labelControl(input, options);
    input.min = options.min ?? 0;
    input.max = options.max ?? 1;
    input.step = options.step ?? 0.05;
    input.value = options.value ?? 0;
    const readout = document.createElement('output');
    readout.className = 'range-out';
    readout.textContent = formatRange(input.value);
    input.addEventListener('input', () => {
        readout.textContent = formatRange(input.value);
    });
    input.addEventListener('change', () => options.onChange(Number(input.value)));
    control.append(input, readout);
    return row;
}

function formatRange(value) {
    return Number(value).toFixed(2);
}

/* ---------------------------------------------------------------- 替换 / 插入弹窗 */

/** 弹出已注册算子列表；层内节点可跨类别替换，后端负责参数校验。 */
export function openReplaceModal(node) {
    if (!node || node.type === 'layer_group') return;
    const candidates = candidateOperators(node);
    openOperatorPicker({
        title: '更换算子',
        subtitle: `当前为 ${node.label ?? node.type}。兼容的同名参数会保留，其余使用适配当前模型尺寸的默认值。`,
        candidates,
        emptyHint: '没有可替换的已注册算子。',
        buildCard: (item, dismiss) => buildOperatorCard(item, node, dismiss),
    });
}

/**
 * 层内插入算子选择器：复用替换弹窗骨架与搜索框。
 * 候选 = 所有允许出现在层内的已注册算子，
 * 选定后调 insertLayerOperation(groupId, afterOperationId, type)。
 * @param {{groupId: string, afterOperationId?: string|null}} options
 */
export function openInsertPicker({ groupId, afterOperationId = null }) {
    const group = (AppState.flowchart?.nodes ?? []).find((node) => node.id === groupId);
    const anchor = afterOperationId == null
        ? null
        : (group?.children ?? []).find((child) => (child.operation_id ?? child.role) === afterOperationId);
    const groupName = group?.label ?? groupId;
    const subtitle = anchor
        ? `层组「${groupName}」· 插入到 ${anchor.label ?? anchor.type ?? afterOperationId} 之后`
        : `层组「${groupName}」· 插入到组末尾`;
    openOperatorPicker({
        title: '添加算子',
        subtitle,
        candidates: insertCandidates(),
        emptyHint: '没有可插入的算子。',
        buildCard: (item, dismiss) => buildInsertCard(item, () => {
            const child = insertLayerOperation(groupId, afterOperationId, item.type);
            if (child) {
                Bus.emit(EVENTS.TOAST, {
                    type: 'success',
                    message: `已插入「${item.chinese_name ?? item.type}」`,
                });
            }
            dismiss();
        }),
    });
}

/** 替换 / 插入共用的弹窗骨架：标题 + 搜索框 + 候选网格（+ 可选底部引导提示） */
function openOperatorPicker({ title, subtitle, candidates, emptyHint, buildCard, footerHint = '' }) {
    const root = document.getElementById('modal-root');
    if (!root) return;

    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop';

    const modal = document.createElement('div');
    modal.className = 'modal';

    const head = document.createElement('header');
    head.className = 'modal-head';
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    const sub = document.createElement('p');
    sub.className = 'hint';
    sub.textContent = subtitle;
    const close = document.createElement('button');
    close.className = 'icon-btn';
    close.textContent = '✕';
    head.append(titleEl, sub, close);

    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'ctl ctl-input modal-search';
    search.placeholder = '搜索算子名称或类型…';

    const list = document.createElement('div');
    list.className = 'op-list';

    const renderList = (keyword = '') => {
        list.innerHTML = '';
        const lower = keyword.trim().toLowerCase();
        const filtered = candidates.filter((item) => !lower
            || item.type.toLowerCase().includes(lower)
            || String(item.chinese_name ?? '').toLowerCase().includes(lower));
        if (!filtered.length) {
            list.appendChild(hintEl(candidates.length ? '没有匹配的算子。' : emptyHint));
            return;
        }
        for (const item of filtered) {
            list.appendChild(buildCard(item, dismiss));
        }
    };

    const dismiss = () => {
        backdrop.classList.add('is-leaving');
        setTimeout(() => backdrop.remove(), 140);
        document.removeEventListener('keydown', onKey);
    };
    const onKey = (event) => {
        if (event.key === 'Escape') dismiss();
    };

    close.addEventListener('click', dismiss);
    backdrop.addEventListener('mousedown', (event) => {
        if (event.target === backdrop) dismiss();
    });
    document.addEventListener('keydown', onKey);
    search.addEventListener('input', () => renderList(search.value));

    modal.append(head, search, list);
    if (footerHint) modal.appendChild(hintEl(footerHint));
    backdrop.appendChild(modal);
    root.appendChild(backdrop);
    renderList();
    search.focus();
}

/** 单个候选算子卡片 */
function buildOperatorCard(item, node, done) {
    const card = operatorCardShell(item);
    if (item.type === node.type) {
        card.classList.add('is-current');
        const tag = document.createElement('em');
        tag.className = 'op-current';
        tag.textContent = '当前';
        card.appendChild(tag);
    }
    card.addEventListener('click', () => {
        if (item.type !== node.type) {
            updateNodeType(node.id, item.type);
            Bus.emit(EVENTS.TOAST, {
                type: 'success',
                message: `已替换为「${item.chinese_name ?? item.type}」`,
            });
        }
        done();
    });
    return card;
}

/** 插入候选卡片：中文名 + 类型代码 + 实现列表，点击即插入 */
function buildInsertCard(item, onPick) {
    const card = operatorCardShell(item);
    card.addEventListener('click', onPick);
    return card;
}

/** 候选算子卡片外壳（替换 / 插入共用） */
function operatorCardShell(item) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = `op-card cat-${categoryOf({ type: item.type, slot: item.slot })}`;
    const name = document.createElement('strong');
    name.textContent = item.chinese_name ?? item.type;
    const code = document.createElement('code');
    code.textContent = item.type;
    const impl = document.createElement('span');
    impl.className = 'op-impl';
    impl.textContent = (item.implementations ?? []).join(' / ') || '默认实现';
    card.append(name, code, impl);
    return card;
}

/** 候选池：优先用 catalog，退化用 schemas */
function operatorPool() {
    const catalog = AppState.operatorCatalog?.operators;
    return Array.isArray(catalog) && catalog.length
        ? catalog
        : Object.entries(AppState.operatorSchemas ?? {}).map(([type, schema]) => ({
            type,
            slot: schema?.slot,
            chinese_name: schema?.chinese_name,
            implementations: schema?.implementations,
        }));
}

/** 收集候选算子。层内有序算子可以跨类别替换。 */
function candidateOperators(node) {
    const slot = node?.slot;
    return operatorPool().filter((item) => isCandidate(item, node, slot));
}

/** 插入候选：与 isCandidate 的 generic（operation_id）分支一致 */
const INSERT_SLOTS = ['any', 'norm', 'attention', 'ffn', 'residual'];
function insertCandidates() {
    return operatorPool().filter((item) => INSERT_SLOTS.includes(item.slot));
}

/**
 * 候选规则：
 *   — slot 为 any 的算子（如 unmodeled）在任何槽位都可选；
 *   — output 槽位内 lm_head 与 sampling 不可互换：后端按 role 回填 head / sampling 两个配置项，
 *     互换会直接产出无法求解的 config。
 */
function isCandidate(item, node, slot) {
    if (node?.operation_id) {
        return INSERT_SLOTS.includes(item.slot);
    }
    if (item.slot === 'any') return true;
    if (slot && item.slot !== slot) return false;
    const pinned = ROLE_PINNED_TYPE[node?.role];
    return !pinned || item.type === pinned;
}

/** role → 只允许的算子类型（与后端 OUTPUT_LAYOUT 保持一致） */
const ROLE_PINNED_TYPE = {
    output_head: 'lm_head',
    output_sampling: 'sampling',
};

/* ---------------------------------------------------------------- 小工具 */

export function sectionEl(titleText) {
    const section = document.createElement('section');
    section.className = 'form-section';
    const title = document.createElement('h4');
    title.className = 'section-title';
    title.textContent = titleText;
    section.appendChild(title);
    return section;
}

export function hintEl(textContent) {
    const p = document.createElement('p');
    p.className = 'hint';
    p.textContent = textContent;
    return p;
}

function metaRow(label, value) {
    const row = document.createElement('div');
    row.className = 'meta-row';
    const k = document.createElement('span');
    k.textContent = label;
    const v = document.createElement('code');
    v.textContent = String(value);
    row.append(k, v);
    return row;
}

/** 由 min/max/default 生成取值范围提示 */
function buildHint(spec) {
    if (!spec) return '';
    const parts = [];
    if (spec.hint) parts.push(spec.hint);
    if (spec.description) parts.push(spec.description);
    if (spec.inherit_from) parts.push(`留空继承 ${spec.inherit_from}`);
    const range = [];
    if (spec.min != null) range.push(`≥ ${spec.min}`);
    if (spec.max != null) range.push(`≤ ${spec.max}`);
    if (range.length) parts.push(`取值 ${range.join('，')}`);
    if (spec.default != null && spec.default !== '' && typeof spec.default !== 'object') {
        parts.push(`默认 ${spec.default}`);
    }
    return parts.join(' · ');
}

/** 枚举选项：后端用空字符串表示“不覆盖”，给个可读的展示名 */
function optionOf(choice) {
    return choice === '' ? { value: '', label: '（跟随模型）' } : choice;
}

/** schema 缺失时按当前值推断控件类型 */
function inferSpec(value) {
    if (typeof value === 'boolean') return { type: 'bool' };
    if (typeof value === 'number') return { type: Number.isInteger(value) ? 'int' : 'float' };
    return { type: 'string' };
}

function clampToSpec(value, spec) {
    let result = value;
    if (spec?.min != null) result = Math.max(spec.min, result);
    if (spec?.max != null) result = Math.min(spec.max, result);
    return result;
}
