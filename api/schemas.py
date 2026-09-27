from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Unique transaction ID")
    timestamp: Optional[str] = Field(default=None, description="ISO-8601 transaction timestamp")
    user_id: str = Field(..., description="Customer ID, e.g. USR_00123")
    card_id: str = Field(..., description="Card reference, e.g. CARD_00123")
    merchant_id: Optional[str] = Field(default="MERCH_0001", description="Merchant ID")
    mcc: str = Field(default="5411", description="Merchant Category Code (e.g. 5411, 5732, 6051)")
    amount: float = Field(..., ge=0.01, description="Transaction amount in USD")
    channel: str = Field(default="web", description="Payment channel: web, mobile, pos, wire_transfer")
    device_id: str = Field(..., description="Hardware device fingerprint")
    ip_address: str = Field(..., description="Originating IPv4 address")
    location_lat: float = Field(default=40.7128, description="Latitude")
    location_lon: float = Field(default=-74.0060, description="Longitude")
    country: Optional[str] = Field(default="US", description="Country ISO code")


class RuleTriggeredDetail(BaseModel):
    rule_id: str
    name: str
    severity: str
    points: float
    description: str


class RiskFactorDetail(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    actual_value: float
    is_risk_increasing: bool


class ScoreResponse(BaseModel):
    transaction_id: str
    composite_risk_score: float
    risk_tier: str
    recommended_action: str
    case_id: Optional[str] = None
    component_scores: Dict[str, float]
    rules_triggered: List[RuleTriggeredDetail]
    top_risk_factors: List[RiskFactorDetail]
    explanation_narrative: str
    processing_time_ms: float


class CaseUpdateRequest(BaseModel):
    new_status: str = Field(..., description="One of: ASSIGNED, UNDER_INVESTIGATION, CONFIRMED_FRAUD, FALSE_POSITIVE, CLEARED")
    actor: str = Field(default="Investigator", description="Name or role of person updating the case")
    notes: str = Field(default="", description="Investigator notes or rationale")
    assigned_investigator: Optional[str] = Field(default=None, description="Reassign to investigator")


class SARResponse(BaseModel):
    case_id: str
    narrative_markdown: str
    pdf_filename: str
    generated_at: str
