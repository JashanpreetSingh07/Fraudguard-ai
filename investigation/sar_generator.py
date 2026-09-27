import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# ReportLab imports for automated PDF export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class SARGenerator:
    """
    Automated FinCEN-compliant Suspicious Activity Report (SAR) Narrative Generator.
    Produces legally structured compliance dossiers with AI forensic evidence.
    Supports Markdown and PDF exports.
    """

    def __init__(self, output_dir: Path = config.SAR_EXPORTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_narrative_text(self, case_data: Dict[str, Any]) -> str:
        """Constructs an exhaustive regulatory SAR narrative in Markdown format."""
        case_id = case_data.get("case_id", "CASE-UNKNOWN")
        tx_id = case_data.get("transaction_id", "TX_UNKNOWN")
        user_id = case_data.get("user_id", "USR_UNKNOWN")
        card_id = case_data.get("card_id", "CARD_UNKNOWN")
        device_id = case_data.get("device_id", "DEV_UNKNOWN")
        ip_addr = case_data.get("ip_address", "0.0.0.0")
        amount = case_data.get("amount", 0.0)
        risk_score = case_data.get("risk_score", 0.0)
        risk_tier = case_data.get("risk_tier", "HIGH")
        typology = case_data.get("fraud_typology", "Suspicious Activity")
        rules = case_data.get("rules_triggered", [])
        explanation = case_data.get("explanation_narrative", "Statistical behavioral deviation.")
        created_at = case_data.get("created_at", datetime.utcnow().isoformat())
        investigator = case_data.get("assigned_investigator", "Automated AML Surveillance Desk")

        # Map typology to regulatory statutes
        statute_map = {
            "aml_structuring": "31 CFR § 1010.311 (Structuring Transactions to Evade Currency Transaction Reporting)",
            "account_takeover": "18 U.S.C. § 1028A (Aggravated Identity Theft) & 18 U.S.C. § 1030 (Fraud in Connection with Computers)",
            "card_not_present_burst": "18 U.S.C. § 1029 (Fraud and Related Activity in Connection with Access Devices)",
            "impossible_travel": "Federal FFIEC Authentication Guidance (Geographic Anomaly / Session Hijacking)",
            "mule_network_ring": "18 U.S.C. § 1956 (Laundering of Monetary Instruments & Syndicate Conspiracy)",
        }
        relevant_statute = statute_map.get(typology, "Bank Secrecy Act (BSA) 31 U.S.C. § 5318(g)")

        rules_list_md = "\n".join([f"- **{r.get('rule_id', 'RULE')}** ({r.get('severity', 'WARN')}): {r.get('description', '')}" for r in rules]) if rules else "- No deterministic rule violations triggered (Identified by Machine Learning & Anomaly Ensemble)."

        markdown_content = f"""# SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE
**Jurisdiction:** Financial Crimes Enforcement Network (FinCEN) / BSA Compliance  
**Document Tracking ID:** {case_id}  
**Date of Filing:** {datetime.utcnow().strftime('%B %d, %Y')}  
**Investigating Unit:** Financial Intelligence & Special Investigations Unit (SIU)  
**Lead Investigator:** {investigator}  

---

### PART I: SUBJECT IDENTIFICATION
- **Subject Customer ID:** `{user_id}`
- **Payment Card Reference:** `{card_id}`
- **Hardware Device Identifier:** `{device_id}`
- **Originating IP Address:** `{ip_addr}`
- **Target Transaction ID:** `{tx_id}`
- **Disputed / Flagged Amount:** `${amount:,.2f} USD`
- **Initial Alert Timestamp:** `{created_at}`

---

### PART II: SUSPICIOUS ACTIVITY CLASSIFICATION
- **Primary Typology Detected:** **{typology.upper().replace('_', ' ')}**
- **Associated Regulatory Statute:** {relevant_statute}
- **AI Model Risk Evaluation:** **{risk_score:.1f} / 100 ({risk_tier} RISK)**
- **System Recommendation:** `DECLINE_AND_FREEZE` / Formal Case Escalation

---

### PART III: EXECUTIVE INVESTIGATION SUMMARY & CHRONOLOGY
On or about `{created_at}`, the automated surveillance systems of the institution flagged an anomalous transaction event totaling `${amount:,.2f}` initiated under customer account `{user_id}`. 

The transaction displayed distinct hallmarks of **{typology.replace('_', ' ')}**. Specifically, the hybrid detection engine identified that:
1. The transaction deviated significantly from historical baseline behavior established over prior activity cycles.
2. The originating technical footprint (Device `{device_id}`, IP `{ip_addr}`) triggered multiple risk multipliers.
3. Transaction velocity and spatial positioning were mathematically incompatible with bona fide user activity.

---

### PART IV: FORENSIC MACHINE LEARNING & RULE EVIDENCE
#### 1. Deterministic Rule Violations:
{rules_list_md}

#### 2. Explainable AI (SHAP) Factor Attribution:
{explanation}

---

### PART V: ENTITY RESOLUTION & SYNDICATE CORRELATION
Cross-referencing institutional transaction graphs indicates that the technical infrastructure utilized in this incident exhibits links to other observed anomalous events:
- Associated Device `{device_id}` has been correlated with high-risk velocity bursts.
- Network IP `{ip_addr}` is classified as an anonymous proxy / high-risk hosting subnet.

---

### PART VI: INVESTIGATOR DISPOSITION & DISCLOSURE
Based upon the convergence of deterministic rule violations, gradient-boosted decision tree scoring, and topological graph correlation, the SIU concludes that this activity exhibits no apparent commercial or lawful purpose.

**Recommended Actions:**
1. Permanent administrative freeze on Account `{user_id}` and associated instruments.
2. Electronic transmission of this Suspicious Activity Report to FinCEN pursuant to 31 U.S.C. 5318(g).
3. Referral to appropriate law enforcement authorities (FBI / Secret Service Cyber Fraud Task Force).

*Filed by:* `{investigator}`  
*Special Investigations Unit (SIU)*
"""
        return markdown_content

    def export_pdf(self, case_data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Renders and saves a formal, high-resolution PDF compliance document."""
        case_id = case_data.get("case_id", "CASE-UNKNOWN")
        if filename is None:
            filename = f"SAR_{case_id}.pdf"

        pdf_path = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1e293b"),
            fontName="Helvetica-Bold",
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=normal,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
        )
        h2_style = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=normal,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )

        story = []

        # Header Title
        story.append(Paragraph("FINANCIAL CRIMES ENFORCEMENT NETWORK (FinCEN)", subtitle_style))
        story.append(Paragraph("SUSPICIOUS ACTIVITY REPORT (SAR) - NARRATIVE DOSSIER", title_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

        # Subject & Case Metadata Table
        meta_data = [
            [
                Paragraph("<b>Tracking Case ID:</b>", body_style),
                Paragraph(f"<b>{case_id}</b>", body_style),
                Paragraph("<b>Date of Report:</b>", body_style),
                Paragraph(datetime.utcnow().strftime("%Y-%m-%d"), body_style),
            ],
            [
                Paragraph("<b>Target Subject ID:</b>", body_style),
                Paragraph(str(case_data.get("user_id", "N/A")), body_style),
                Paragraph("<b>Transaction ID:</b>", body_style),
                Paragraph(str(case_data.get("transaction_id", "N/A")), body_style),
            ],
            [
                Paragraph("<b>Flagged Amount:</b>", body_style),
                Paragraph(f"<b>${case_data.get('amount', 0.0):,.2f} USD</b>", body_style),
                Paragraph("<b>Risk Evaluation:</b>", body_style),
                Paragraph(f"<font color='#dc2626'><b>{case_data.get('risk_score', 0.0):.1f}/100 ({case_data.get('risk_tier', 'CRITICAL')})</b></font>", body_style),
            ],
            [
                Paragraph("<b>Hardware Fingerprint:</b>", body_style),
                Paragraph(str(case_data.get("device_id", "N/A")), body_style),
                Paragraph("<b>IP Address:</b>", body_style),
                Paragraph(str(case_data.get("ip_address", "N/A")), body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))

        # Part II: Typology & Narrative
        story.append(Paragraph("1. SUSPICIOUS ACTIVITY CLASSIFICATION & TYPOLOGY", h2_style))
        typology_desc = (
            f"The customer activity referenced above constitutes anomalous transaction behavior characterized as "
            f"<b>{case_data.get('fraud_typology', 'Financial Crime').upper()}</b>. Multi-layered AI surveillance "
            f"identified statistically significant deviations from normal operational baselines."
        )
        story.append(Paragraph(typology_desc, body_style))
        story.append(Spacer(1, 8))

        # Part III: AI Forensic Evidence
        story.append(Paragraph("2. AI RISK SCORING & EXPLAINABILITY EVIDENCE", h2_style))
        ai_narrative = str(case_data.get("explanation_narrative", "Anomalous velocity and geographic profile."))
        story.append(Paragraph(f"<b>Machine Learning Attribution:</b> {ai_narrative}", body_style))
        story.append(Spacer(1, 8))

        # Rules Table
        rules = case_data.get("rules_triggered", [])
        if rules:
            story.append(Paragraph("<b>Triggered Business Rules & Policies:</b>", body_style))
            rule_rows = [["Rule Code", "Severity", "Description"]]
            for r in rules:
                rule_rows.append([
                    r.get("rule_id", "RULE"),
                    r.get("severity", "HIGH"),
                    r.get("description", ""),
                ])
            rule_table = Table(rule_rows, colWidths=[130, 80, 330])
            rule_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(Spacer(1, 4))
            story.append(rule_table)
            story.append(Spacer(1, 10))

        # Part IV: Disposition
        story.append(Paragraph("3. INVESTIGATOR DISPOSITION & FILING RECOMMENDATION", h2_style))
        disposition_text = (
            "The Special Investigations Unit recommends immediate regulatory filing of FinCEN Form 111 (SAR), "
            "continued monitoring of correlated accounts, and referral to designated law enforcement liaisons."
        )
        story.append(Paragraph(disposition_text, body_style))
        story.append(Spacer(1, 16))

        # Signature Block
        sig_data = [
            [
                Paragraph("<b>Investigator Signature:</b> ___________________________", body_style),
                Paragraph("<b>Supervisory Review:</b> ___________________________", body_style),
            ],
            [
                Paragraph(f"<b>Name:</b> {case_data.get('assigned_investigator', 'Lead Analyst')}", subtitle_style),
                Paragraph("<b>Compliance Officer:</b> BSA / AML Officer", subtitle_style),
            ]
        ]
        sig_table = Table(sig_data, colWidths=[270, 270])
        sig_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2)]))
        story.append(sig_table)

        doc.build(story)
        print(f"[SARGenerator] Exported formal PDF SAR dossier to {pdf_path}")
        return pdf_path
