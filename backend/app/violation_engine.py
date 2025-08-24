"""
Billboard Violation Detection Engine
Implements rule-based compliance checking using rules.yml configuration
"""

import yaml
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ViolationEngine:
    """Rule-based billboard violation detection system"""
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize violation engine with rules configuration
        
        Args:
            rules_path: Path to rules.yml file
        """
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "..", "rules.yml")
        
        self.rules = self._load_rules(rules_path)
        self.violation_history = []
        
    def _load_rules(self, rules_path: str) -> Dict:
        """Load violation rules from YAML configuration"""
        try:
            with open(rules_path, 'r') as file:
                rules = yaml.safe_load(file)
                logger.info(f"Loaded violation rules from {rules_path}")
                return rules
        except Exception as e:
            logger.error(f"Failed to load rules from {rules_path}: {e}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Fallback default rules if YAML loading fails"""
        return {
            'billboard_regulations': {
                'size_limits': {
                    'residential_zone': {'max_area_sqm': 10.0, 'max_width_m': 4.0, 'max_height_m': 3.0},
                    'commercial_zone': {'max_area_sqm': 25.0, 'max_width_m': 8.0, 'max_height_m': 4.0}
                },
                'placement_restrictions': {
                    'traffic_junction_distance': {'minimum_meters': 100.0},
                    'traffic_signal_distance': {'minimum_meters': 50.0}
                }
            }
        }
    
    def check_violations(self, billboard_data: Dict) -> Dict:
        """
        Check billboard for regulatory violations
        
        Args:
            billboard_data: Dict containing billboard information
            
        Returns:
            Dict with violation results
        """
        violations = []
        compliance_score = 100.0
        
        try:
            # Check size violations
            size_violations = self._check_size_violations(billboard_data)
            violations.extend(size_violations)
            
            # Check placement violations
            placement_violations = self._check_placement_violations(billboard_data)
            violations.extend(placement_violations)
            
            # Check licensing violations
            license_violations = self._check_licensing_violations(billboard_data)
            violations.extend(license_violations)
            
            # Check content violations (if content analysis available)
            content_violations = self._check_content_violations(billboard_data)
            violations.extend(content_violations)
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(violations)
            
            # Determine enforcement action
            enforcement = self._determine_enforcement_action(violations)
            
            return {
                'violations': violations,
                'violation_count': len(violations),
                'compliance_score': compliance_score,
                'enforcement_action': enforcement,
                'inspection_required': len(violations) > 0,
                'timestamp': datetime.now().isoformat(),
                'billboard_id': billboard_data.get('id', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Violation check failed: {e}")
            return {
                'error': str(e),
                'violations': [],
                'compliance_score': 0.0
            }
    
    def _check_size_violations(self, billboard_data: Dict) -> List[Dict]:
        """Check for size limit violations"""
        violations = []
        
        try:
            dimensions = billboard_data.get('dimensions', {})
            zone_type = billboard_data.get('zone_type', 'commercial_zone')
            
            if not dimensions:
                return violations
            
            width = dimensions.get('width_meters', 0)
            height = dimensions.get('height_meters', 0)
            area = dimensions.get('area_sqm', width * height)
            
            # Get size limits for zone
            size_limits = self.rules['billboard_regulations']['size_limits'].get(zone_type, {})
            
            if not size_limits:
                return violations
            
            # Check area violation
            max_area = size_limits.get('max_area_sqm', float('inf'))
            if area > max_area:
                violations.append({
                    'type': 'size_violation',
                    'subtype': 'area_exceeded',
                    'severity': 'major',
                    'description': f"Billboard area {area:.1f}m² exceeds limit of {max_area}m² for {zone_type}",
                    'measured_value': area,
                    'limit_value': max_area,
                    'zone_type': zone_type
                })
            
            # Check width violation
            max_width = size_limits.get('max_width_m', float('inf'))
            if width > max_width:
                violations.append({
                    'type': 'size_violation',
                    'subtype': 'width_exceeded',
                    'severity': 'major',
                    'description': f"Billboard width {width:.1f}m exceeds limit of {max_width}m for {zone_type}",
                    'measured_value': width,
                    'limit_value': max_width,
                    'zone_type': zone_type
                })
            
            # Check height violation
            max_height = size_limits.get('max_height_m', float('inf'))
            if height > max_height:
                violations.append({
                    'type': 'size_violation',
                    'subtype': 'height_exceeded',
                    'severity': 'major',
                    'description': f"Billboard height {height:.1f}m exceeds limit of {max_height}m for {zone_type}",
                    'measured_value': height,
                    'limit_value': max_height,
                    'zone_type': zone_type
                })
                
        except Exception as e:
            logger.error(f"Size violation check failed: {e}")
            
        return violations
    
    def _check_placement_violations(self, billboard_data: Dict) -> List[Dict]:
        """Check for placement restriction violations"""
        violations = []
        
        try:
            junction_distance = billboard_data.get('junction_distance_meters')
            placement_rules = self.rules['billboard_regulations']['placement_restrictions']
            
            # Check traffic junction distance
            if junction_distance is not None:
                min_junction_distance = placement_rules['traffic_junction_distance']['minimum_meters']
                if junction_distance < min_junction_distance:
                    violations.append({
                        'type': 'placement_violation',
                        'subtype': 'junction_too_close',
                        'severity': 'critical',
                        'description': f"Billboard is {junction_distance:.1f}m from traffic junction, minimum required: {min_junction_distance}m",
                        'measured_value': junction_distance,
                        'limit_value': min_junction_distance,
                        'reason': placement_rules['traffic_junction_distance'].get('reason', 'Safety requirement')
                    })
            
            # Check other placement restrictions
            gps_coords = billboard_data.get('gps_coordinates')
            if gps_coords:
                # In production, would check against:
                # - School zone databases
                # - Hospital locations
                # - Residential area boundaries
                # - Traffic signal locations
                pass
                
        except Exception as e:
            logger.error(f"Placement violation check failed: {e}")
            
        return violations
    
    def _check_licensing_violations(self, billboard_data: Dict) -> List[Dict]:
        """Check for licensing and permit violations"""
        violations = []
        
        try:
            permit_info = billboard_data.get('permit_info', {})
            licensing_rules = self.rules['billboard_regulations']['licensing_requirements']
            
            # Check if permit is required and present
            if licensing_rules.get('permit_required', True):
                permit_number = permit_info.get('permit_number')
                if not permit_number:
                    violations.append({
                        'type': 'licensing_violation',
                        'subtype': 'missing_permit',
                        'severity': 'major',
                        'description': "Billboard does not display required permit number",
                        'required_action': 'Obtain and display valid permit'
                    })
                
                # Check permit expiration
                expiry_date = permit_info.get('expiry_date')
                if expiry_date:
                    try:
                        expiry = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                        if expiry < datetime.now():
                            violations.append({
                                'type': 'licensing_violation',
                                'subtype': 'expired_permit',
                                'severity': 'major',
                                'description': f"Billboard permit expired on {expiry_date}",
                                'expiry_date': expiry_date,
                                'required_action': 'Renew permit immediately'
                            })
                    except ValueError:
                        pass
                        
        except Exception as e:
            logger.error(f"Licensing violation check failed: {e}")
            
        return violations
    
    def _check_content_violations(self, billboard_data: Dict) -> List[Dict]:
        """Check for content restriction violations"""
        violations = []
        
        try:
            content_analysis = billboard_data.get('content_analysis', {})
            content_rules = self.rules['billboard_regulations'].get('content_restrictions', {})
            
            # Check for prohibited content
            prohibited_content = content_analysis.get('detected_content', [])
            prohibited_list = content_rules.get('prohibited_content', [])
            
            for content_type in prohibited_content:
                if content_type in prohibited_list:
                    violations.append({
                        'type': 'content_violation',
                        'subtype': 'prohibited_content',
                        'severity': 'critical',
                        'description': f"Billboard contains prohibited content: {content_type}",
                        'detected_content': content_type,
                        'required_action': 'Remove or modify content immediately'
                    })
                    
        except Exception as e:
            logger.error(f"Content violation check failed: {e}")
            
        return violations
    
    def _calculate_compliance_score(self, violations: List[Dict]) -> float:
        """Calculate overall compliance score (0-100)"""
        if not violations:
            return 100.0
        
        penalty_weights = {
            'critical': 30.0,
            'major': 15.0,
            'minor': 5.0
        }
        
        total_penalty = 0.0
        for violation in violations:
            severity = violation.get('severity', 'minor')
            penalty = penalty_weights.get(severity, 5.0)
            total_penalty += penalty
        
        compliance_score = max(0.0, 100.0 - total_penalty)
        return compliance_score
    
    def _determine_enforcement_action(self, violations: List[Dict]) -> Dict:
        """Determine appropriate enforcement action based on violations"""
        if not violations:
            return {'action': 'none', 'description': 'No violations detected'}
        
        # Find highest severity violation
        severities = [v.get('severity', 'minor') for v in violations]
        
        if 'critical' in severities:
            enforcement_rules = self.rules.get('enforcement_actions', {}).get('critical', {})
            return {
                'action': 'immediate_removal',
                'severity': 'critical',
                'immediate_removal': enforcement_rules.get('immediate_removal', True),
                'fine_amount_usd': enforcement_rules.get('fine_amount_usd', 5000),
                'permit_suspension': enforcement_rules.get('permit_suspension', True),
                'legal_action': enforcement_rules.get('legal_action', True),
                'description': 'Critical violations require immediate enforcement action'
            }
        elif 'major' in severities:
            enforcement_rules = self.rules.get('enforcement_actions', {}).get('major', {})
            return {
                'action': 'notice_and_correction',
                'severity': 'major',
                'notice_period_days': enforcement_rules.get('notice_period_days', 7),
                'fine_amount_usd': enforcement_rules.get('fine_amount_usd', 1500),
                'correction_required': enforcement_rules.get('correction_required', True),
                'reinspection_required': enforcement_rules.get('reinspection_required', True),
                'description': 'Major violations require correction within notice period'
            }
        else:
            enforcement_rules = self.rules.get('enforcement_actions', {}).get('minor', {})
            return {
                'action': 'warning',
                'severity': 'minor',
                'notice_period_days': enforcement_rules.get('notice_period_days', 30),
                'fine_amount_usd': enforcement_rules.get('fine_amount_usd', 300),
                'warning_issued': enforcement_rules.get('warning_issued', True),
                'voluntary_compliance': enforcement_rules.get('voluntary_compliance', True),
                'description': 'Minor violations can be resolved through voluntary compliance'
            }


def create_violation_engine():
    """Factory function to create violation engine"""
    return ViolationEngine()
