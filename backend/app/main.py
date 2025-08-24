
import os
import logging
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional

# Import existing modules
from .db import init_db
from .routers import reports, registry, review, stats

# Import new hackathon modules
from .detection import create_inference_pipeline
from .depth_estimation import create_measurement_pipeline
from .violation_engine import create_violation_engine
from .privacy import create_privacy_manager
from .auth import create_auth_manager, get_current_user, get_current_citizen, UserRole, Permission
from .security import create_rate_limiter, create_security_middleware, create_file_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Billboard Sentinel - AI-Powered Civic Enforcement",
    version="2.0.0",
    description="Complete billboard detection and violation management system with AI, privacy protection, and RBAC",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize security components
rate_limiter = create_rate_limiter()
security_middleware = create_security_middleware(app, rate_limiter)
app.add_middleware(security_middleware.__class__, rate_limiter=rate_limiter)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize components
try:
    init_db()
    detection_pipeline = create_inference_pipeline()
    measurement_pipeline = create_measurement_pipeline()
    violation_engine = create_violation_engine()
    privacy_manager = create_privacy_manager()
    auth_manager = create_auth_manager()
    file_validator = create_file_validator()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Component initialization failed: {e}")

# Include existing routers
app.include_router(reports.router, prefix="/api")
app.include_router(registry.router, prefix="/api")
app.include_router(review.router, prefix="/api")
app.include_router(stats.router, prefix="/api")

# Pydantic models for new endpoints
class LoginRequest(BaseModel):
    email: str
    password: str

class DetectionRequest(BaseModel):
    image_path: str
    gps_coordinates: Optional[tuple] = None
    zone_type: str = "commercial_zone"

class PrivacyConsentRequest(BaseModel):
    face_blur_consent: bool
    data_sharing_consent: bool = False
    municipality_sharing: bool = False
    retention_acknowledged: bool
    user_id: Optional[str] = None

# New API endpoints for hackathon features

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Authenticate user and return JWT token"""
    try:
        user = auth_manager.authenticate_user(request.email, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = auth_manager.create_access_token(user)
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_info": {
                "user_id": user["user_id"],
                "email": user["email"],
                "role": user["role"].value if hasattr(user["role"], "value") else user["role"],
                "name": user["name"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "role": current_user["role"],
        "name": current_user["name"]
    }

@app.post("/api/detection/analyze")
async def analyze_billboard(
    file: UploadFile = File(...),
    gps_lat: float = Form(...),
    gps_lon: float = Form(...),
    zone_type: str = Form("commercial_zone"),
    consent_data: str = Form(...),
    current_user: dict = Depends(get_current_citizen)
):
    """Complete billboard analysis pipeline with AI detection"""
    try:
        import json
        consent = json.loads(consent_data)
        
        # Validate file upload
        validation = await file_validator.validate_upload(file)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid file: {validation['errors']}")
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Process with privacy protection
            privacy_result = privacy_manager.process_image_with_privacy(temp_path, consent)
            processed_image_path = privacy_result.get("processed_path", temp_path)
            
            # Run AI detection
            detection_result = detection_pipeline.detect_billboards(processed_image_path)
            
            # Measure dimensions if billboards detected
            measurement_results = []
            if detection_result["detection_count"] > 0:
                measurement_pipeline_components = measurement_pipeline
                depth_estimator = measurement_pipeline_components["depth_estimator"]
                
                for detection in detection_result["detections"]:
                    measurement = depth_estimator.measure_billboard_dimensions(
                        processed_image_path, 
                        detection["bbox"],
                        (gps_lat, gps_lon)
                    )
                    measurement_results.append(measurement)
            
            # Check violations for each detected billboard
            violation_results = []
            for i, detection in enumerate(detection_result["detections"]):
                billboard_data = {
                    "dimensions": measurement_results[i]["dimensions"] if i < len(measurement_results) else {},
                    "zone_type": zone_type,
                    "junction_distance_meters": measurement_results[i].get("junction_distance_meters") if i < len(measurement_results) else None,
                    "gps_coordinates": (gps_lat, gps_lon),
                    "permit_info": {},  # Would be populated from database
                    "detection_confidence": detection["confidence"]
                }
                
                violations = violation_engine.check_violations(billboard_data)
                violation_results.append(violations)
            
            return {
                "analysis_id": f"analysis_{current_user['user_id']}_{int(time.time())}",
                "detection": detection_result,
                "measurements": measurement_results,
                "violations": violation_results,
                "privacy": privacy_result,
                "gps_coordinates": {"latitude": gps_lat, "longitude": gps_lon},
                "zone_type": zone_type,
                "processed_by": current_user["user_id"],
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Billboard analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/privacy/disclaimer")
async def get_privacy_disclaimer():
    """Get privacy disclaimer for UI display"""
    return privacy_manager.get_privacy_disclaimer()

@app.get("/api/violations/rules")
async def get_violation_rules(current_user: dict = Depends(get_current_citizen)):
    """Get violation rules configuration"""
    return violation_engine.rules

@app.get("/api/detection/model-info")
async def get_model_info():
    """Get AI model information and training pipeline details"""
    from .detection import TRAINING_PIPELINE_INFO
    return TRAINING_PIPELINE_INFO

@app.get("/health")
def health_check():
    """Enhanced health check with component status"""
    try:
        component_status = {
            "database": "healthy",
            "ai_detection": "healthy" if detection_pipeline else "unavailable",
            "violation_engine": "healthy" if violation_engine else "unavailable",
            "privacy_manager": "healthy" if privacy_manager else "unavailable",
            "auth_system": "healthy" if auth_manager else "unavailable"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in component_status.values()) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": component_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Import required modules for the new endpoints
import time
from datetime import datetime
