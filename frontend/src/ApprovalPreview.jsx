import React from "react";

const FIELDS = {
  patient_name: "患者", name: "患者", outpatient_no: "患者编号", sex: "性别", gender: "性别",
  age: "年龄", lesion_site: "病灶部位", region: "病灶部位", chief_complaint: "病例描述",
  phone: "联系电话", medical_history: "既往病史", allergy_history: "过敏史",
  case_id: "病例编号", report_id: "报告编号", document_id: "文档编号", title: "标题", filename: "文件名",
  itch: "瘙痒", hurt: "疼痛", bleed: "出血", grew: "近期增大", changed: "近期变化",
  diameter_1: "长径（毫米）", diameter_2: "短径（毫米）", skin_cancer_history: "皮肤癌病史", cancer_history: "其他肿瘤病史",
};
const VALUES = { MALE: "男", FEMALE: "女", ABDOMEN: "腹部", ARM: "上臂", BACK: "背部", CHEST: "胸部", EAR: "耳部", FACE: "面部", FOOT: "足部", FOREARM: "前臂", HAND: "手部", LIP: "唇部", NECK: "颈部", NOSE: "鼻部", SCALP: "头皮", THIGH: "大腿" };
const GROUPS = new Set(["结构化病例", "clinical_metadata", "patient", "patch"]);
const HIDDEN = new Set(["模型", "model", "图片资产", "asset_id", "asset_path"]);

function previewRows(preview) {
  return Object.entries(preview || {}).flatMap(([key, value]) => {
    if (HIDDEN.has(key) || value == null) return [];
    if (value && typeof value === "object") return GROUPS.has(key) ? previewRows(value) : [];
    const label = FIELDS[key] || (/^[\u3400-\u9fff]/.test(key) ? key : "");
    if (!label) return [];
    const text = typeof value === "boolean" ? (value ? "是" : "否") : VALUES[value] || String(value) || "未填写";
    return [[label, text]];
  });
}

export default function ApprovalPreview({ preview }) {
  const rows = previewRows(preview);
  if (!rows.length) return null;
  return <details className="approval-preview"><summary>核对内容</summary><dl>{rows.map(([label, value], index) => <div key={`${label}-${index}`}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></details>;
}
