/** 页面元素和数值读取的公共工具。 */

export const $ = (selector) => document.querySelector(selector);
export const $$ = (selector) => [...document.querySelectorAll(selector)];

export function setValue(id, value) {
  const element = document.getElementById(id);
  if (!element) return;
  if (element.type === "checkbox") element.checked = Boolean(value);
  else element.value = value ?? "";
}

export function numberValue(id) {
  return Number(document.getElementById(id).value);
}

export function optionalNumber(id) {
  const raw = document.getElementById(id).value.trim();
  return raw === "" ? null : Number(raw);
}
