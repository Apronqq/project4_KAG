from __future__ import annotations

from app.schemas.exam import EvidenceChunk


INDICATOR_ALIASES: dict[str, tuple[str, str, str | None]] = {
    "收缩压": ("blood_pressure_systolic", "收缩压", "mmHg"),
    "舒张压": ("blood_pressure_diastolic", "舒张压", "mmHg"),
    "血压": ("blood_pressure", "血压", "mmHg"),
    "空腹血糖": ("fasting_blood_glucose", "空腹血糖", "mmol/L"),
    "fbg": ("fasting_blood_glucose", "空腹血糖", "mmol/L"),
    "hba1c": ("hba1c", "糖化血红蛋白", "%"),
    "糖化血红蛋白": ("hba1c", "糖化血红蛋白", "%"),
    "ldl": ("ldl_c", "低密度脂蛋白胆固醇", "mmol/L"),
    "ldl-c": ("ldl_c", "低密度脂蛋白胆固醇", "mmol/L"),
    "低密度脂蛋白": ("ldl_c", "低密度脂蛋白胆固醇", "mmol/L"),
    "甘油三酯": ("triglycerides", "甘油三酯", "mmol/L"),
    "tg": ("triglycerides", "甘油三酯", "mmol/L"),
    "肌酐": ("creatinine", "肌酐", "umol/L"),
    "cr": ("creatinine", "肌酐", "umol/L"),
    "eGFR": ("egfr", "估算肾小球滤过率", "mL/min/1.73m2"),
    "egfr": ("egfr", "估算肾小球滤过率", "mL/min/1.73m2"),
    "alt": ("alt", "丙氨酸氨基转移酶", "U/L"),
    "谷丙转氨酶": ("alt", "丙氨酸氨基转移酶", "U/L"),
    "ast": ("ast", "天冬氨酸氨基转移酶", "U/L"),
}


STATE_TO_RISK: dict[str, list[dict[str, object]]] = {
    "SBP_high_stage2": [
        {
            "risk_code": "hypertension_risk",
            "risk_name": "高血压风险",
            "disease_code": "hypertension",
            "disease_name": "高血压",
            "risk_level": "high",
            "graph_score": 0.95,
        }
    ],
    "DBP_high_stage2": [
        {
            "risk_code": "hypertension_risk",
            "risk_name": "高血压风险",
            "disease_code": "hypertension",
            "disease_name": "高血压",
            "risk_level": "high",
            "graph_score": 0.90,
        }
    ],
    "FBG_prediabetes": [
        {
            "risk_code": "prediabetes_risk",
            "risk_name": "糖前期风险",
            "disease_code": "prediabetes",
            "disease_name": "糖前期",
            "risk_level": "medium",
            "graph_score": 0.78,
        }
    ],
    "FBG_diabetes": [
        {
            "risk_code": "diabetes_risk",
            "risk_name": "糖尿病风险",
            "disease_code": "type2_diabetes",
            "disease_name": "2型糖尿病",
            "risk_level": "high",
            "graph_score": 0.92,
        }
    ],
    "HbA1c_prediabetes": [
        {
            "risk_code": "prediabetes_risk",
            "risk_name": "糖前期风险",
            "disease_code": "prediabetes",
            "disease_name": "糖前期",
            "risk_level": "medium",
            "graph_score": 0.80,
        }
    ],
    "HbA1c_diabetes": [
        {
            "risk_code": "diabetes_risk",
            "risk_name": "糖尿病风险",
            "disease_code": "type2_diabetes",
            "disease_name": "2型糖尿病",
            "risk_level": "high",
            "graph_score": 0.94,
        }
    ],
    "LDL_high": [
        {
            "risk_code": "dyslipidemia_risk",
            "risk_name": "血脂异常风险",
            "disease_code": "dyslipidemia",
            "disease_name": "血脂异常",
            "risk_level": "medium",
            "graph_score": 0.75,
        }
    ],
    "TG_high": [
        {
            "risk_code": "dyslipidemia_risk",
            "risk_name": "血脂异常风险",
            "disease_code": "dyslipidemia",
            "disease_name": "血脂异常",
            "risk_level": "medium",
            "graph_score": 0.72,
        }
    ],
    "eGFR_moderately_low": [
        {
            "risk_code": "ckd_risk",
            "risk_name": "慢性肾病风险",
            "disease_code": "ckd",
            "disease_name": "慢性肾病",
            "risk_level": "high",
            "graph_score": 0.88,
        }
    ],
    "ALT_high": [
        {
            "risk_code": "liver_function_abnormal_risk",
            "risk_name": "肝功能异常风险",
            "disease_code": "liver_function_abnormality",
            "disease_name": "肝功能异常",
            "risk_level": "medium",
            "graph_score": 0.70,
        }
    ],
    "CREATININE_high": [
        {
            "risk_code": "ckd_risk",
            "risk_name": "慢性肾病风险",
            "disease_code": "ckd",
            "disease_name": "慢性肾病",
            "risk_level": "medium",
            "graph_score": 0.74,
        }
    ],
    "AST_high": [
        {
            "risk_code": "liver_function_abnormal_risk",
            "risk_name": "肝功能异常风险",
            "disease_code": "liver_function_abnormality",
            "disease_name": "肝功能异常",
            "risk_level": "medium",
            "graph_score": 0.68,
        }
    ],
    "BP_stage2_combined": [
        {
            "risk_code": "hypertension_risk",
            "risk_name": "高血压风险",
            "disease_code": "hypertension",
            "disease_name": "高血压",
            "risk_level": "high",
            "graph_score": 0.98,
        }
    ],
    "DIABETES_strong_combined": [
        {
            "risk_code": "diabetes_risk",
            "risk_name": "糖尿病风险",
            "disease_code": "type2_diabetes",
            "disease_name": "2型糖尿病",
            "risk_level": "high",
            "graph_score": 0.98,
        }
    ],
    "PREDIABETES_combined": [
        {
            "risk_code": "prediabetes_risk",
            "risk_name": "糖前期风险",
            "disease_code": "prediabetes",
            "disease_name": "糖前期",
            "risk_level": "medium",
            "graph_score": 0.86,
        }
    ],
    "CKD_strong_combined": [
        {
            "risk_code": "ckd_risk",
            "risk_name": "慢性肾病风险",
            "disease_code": "ckd",
            "disease_name": "慢性肾病",
            "risk_level": "high",
            "graph_score": 0.96,
        }
    ],
    "DYSLIPIDEMIA_combined": [
        {
            "risk_code": "dyslipidemia_risk",
            "risk_name": "血脂异常风险",
            "disease_code": "dyslipidemia",
            "disease_name": "血脂异常",
            "risk_level": "medium",
            "graph_score": 0.82,
        }
    ],
    "ELDERLY_HYPERTENSION_combined": [
        {
            "risk_code": "hypertension_risk",
            "risk_name": "高血压风险",
            "disease_code": "hypertension",
            "disease_name": "高血压",
            "risk_level": "high",
            "graph_score": 0.97,
        }
    ],
}


DISEASE_TO_INTERVENTIONS: dict[str, dict[str, list[str]]] = {
    "hypertension": {
        "interventions": ["限盐", "减重", "规律运动", "居家血压监测"],
        "medication_directions": ["评估是否需要启动降压治疗", "优先结合心内科评估 ACEI/ARB 等方案"],
        "contraindications": [],
        "follow_up_tests": ["动态血压监测", "肾功能复查", "尿常规"],
        "departments": ["心内科", "全科医学科"],
    },
    "prediabetes": {
        "interventions": ["控制总热量摄入", "规律有氧运动", "减重"],
        "medication_directions": ["优先生活方式干预，必要时由内分泌科进一步评估药物干预"],
        "contraindications": [],
        "follow_up_tests": ["空腹血糖复查", "HbA1c 复查"],
        "departments": ["内分泌科"],
    },
    "type2_diabetes": {
        "interventions": ["糖尿病饮食管理", "规律运动", "体重管理"],
        "medication_directions": ["由内分泌科评估降糖治疗方案", "合并肾功能异常时需调整药物选择"],
        "contraindications": ["肾功能下降时部分降糖药需谨慎使用"],
        "follow_up_tests": ["空腹血糖", "HbA1c", "尿白蛋白/肌酐比", "眼底检查"],
        "departments": ["内分泌科"],
    },
    "dyslipidemia": {
        "interventions": ["低脂饮食", "体重管理", "增加运动"],
        "medication_directions": ["由医生评估是否需要他汀类治疗"],
        "contraindications": ["肝功能异常时使用部分降脂药需谨慎"],
        "follow_up_tests": ["血脂复查", "肝功能复查"],
        "departments": ["心内科", "内分泌科"],
    },
    "ckd": {
        "interventions": ["控制血压", "避免肾毒性药物", "低盐饮食"],
        "medication_directions": ["合并高血压时可由医生评估 ACEI/ARB 类药物", "肾功能下降时用药需按 eGFR 调整"],
        "contraindications": ["肾功能不全时部分药物需要减量或禁用"],
        "follow_up_tests": ["肌酐复查", "eGFR 复查", "尿蛋白评估", "肾脏超声"],
        "departments": ["肾内科"],
    },
    "liver_function_abnormality": {
        "interventions": ["戒酒", "控制体重", "复查肝功能"],
        "medication_directions": ["由肝病专科评估是否需要进一步药物或影像检查"],
        "contraindications": ["肝功能异常时部分药物存在肝毒性风险"],
        "follow_up_tests": ["肝功能复查", "腹部超声", "乙肝丙肝筛查"],
        "departments": ["消化内科", "肝病科"],
    },
}


EVIDENCE_CHUNKS: list[EvidenceChunk] = [
    EvidenceChunk(
        chunk_id="guideline_hypertension_adult_2024_1",
        doc_id="guideline_hypertension_adult_2024",
        title="成人高血压管理要点",
        text="成人收缩压和舒张压持续升高提示高血压风险，应结合家庭血压或动态血压进一步评估，并尽早进行生活方式干预。",
        linked_node_codes=["hypertension_risk", "hypertension"],
        source_type="guideline",
        relevance_score=0.90,
    ),
    EvidenceChunk(
        chunk_id="guideline_prediabetes_2024_1",
        doc_id="guideline_diabetes_adult_2024",
        title="糖前期与糖尿病筛查",
        text="空腹血糖或糖化血红蛋白升高提示糖代谢异常，应结合复查和生活方式管理评估糖前期或糖尿病风险。",
        linked_node_codes=["prediabetes_risk", "diabetes_risk", "prediabetes", "type2_diabetes"],
        source_type="guideline",
        relevance_score=0.88,
    ),
    EvidenceChunk(
        chunk_id="guideline_ckd_2024_1",
        doc_id="guideline_ckd_adult_2024",
        title="慢性肾病风险识别",
        text="eGFR 下降和尿蛋白异常是慢性肾病风险的重要线索，需要结合血压、糖代谢和复查结果进一步评估。",
        linked_node_codes=["ckd_risk", "ckd"],
        source_type="guideline",
        relevance_score=0.92,
    ),
    EvidenceChunk(
        chunk_id="guideline_dyslipidemia_2024_1",
        doc_id="guideline_dyslipidemia_adult_2024",
        title="血脂异常干预原则",
        text="低密度脂蛋白胆固醇和甘油三酯升高提示血脂异常风险，首选生活方式干预，必要时由专科评估药物治疗。",
        linked_node_codes=["dyslipidemia_risk", "dyslipidemia"],
        source_type="guideline",
        relevance_score=0.84,
    ),
    EvidenceChunk(
        chunk_id="guideline_liver_function_2024_1",
        doc_id="guideline_liver_function_2024",
        title="肝功能异常评估",
        text="转氨酶升高提示肝功能异常风险，应结合饮酒、脂肪肝和病毒性肝炎因素进一步评估，并复查肝功能。",
        linked_node_codes=["liver_function_abnormal_risk", "liver_function_abnormality"],
        source_type="guideline",
        relevance_score=0.82,
    ),
]
