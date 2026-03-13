#!/usr/bin/env python3
"""
Risk score calculation script
Called from Node.js backend
"""

import sys
import json
from risk_scoring import RiskScoringEngine


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)
    
    try:
        # Parse arguments
        violations_json = sys.argv[1]
        exam_duration = float(sys.argv[2])
        
        violations = json.loads(violations_json)
        
        # Calculate risk score
        engine = RiskScoringEngine()
        risk_score = engine.calculate_risk_score(violations, exam_duration)
        
        # Output as JSON
        print(json.dumps(risk_score))
        sys.exit(0)
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
