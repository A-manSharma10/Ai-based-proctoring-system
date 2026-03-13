const express = require('express');
const { query } = require('../config/database');
const { requireRole } = require('../middleware/auth');
const logger = require('../utils/logger');
const { spawn } = require('child_process');
const path = require('path');

const router = express.Router();

/**
 * Calculate risk score for a session
 */
router.get('/session/:sessionId/score', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    // Get session details
    const sessions = await query(
      'SELECT * FROM exam_sessions WHERE id = ?',
      [sessionId]
    );
    
    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    
    const session = sessions[0];
    
    // Calculate exam duration in minutes
    let examDuration = 0;
    if (session.start_time && session.end_time) {
      examDuration = (new Date(session.end_time) - new Date(session.start_time)) / (1000 * 60);
    } else if (session.start_time) {
      examDuration = (new Date() - new Date(session.start_time)) / (1000 * 60);
    }
    
    // Get all alerts for this session
    const alerts = await query(
      `SELECT alert_type, severity, title, description, confidence_score, metadata, created_at
       FROM alerts 
       WHERE session_id = ? 
       ORDER BY created_at DESC`,
      [sessionId]
    );
    
    // Categorize violations
    const violations = {
      face: [],
      gaze: [],
      object: [],
      audio: []
    };
    
    alerts.forEach(alert => {
      const violation = {
        type: alert.title,
        severity: alert.severity,
        description: alert.description,
        confidence: alert.confidence_score,
        timestamp: alert.created_at
      };
      
      // Parse metadata if available
      if (alert.metadata) {
        try {
          const metadata = typeof alert.metadata === 'string' 
            ? JSON.parse(alert.metadata) 
            : alert.metadata;
          Object.assign(violation, metadata);
        } catch (e) {
          // Ignore parse errors
        }
      }
      
      // Categorize by alert type
      if (alert.alert_type === 'face') {
        violations.face.push(violation);
      } else if (alert.alert_type === 'behavior') {
        // Gaze tracking falls under behavior
        if (alert.title.toLowerCase().includes('gaze') || 
            alert.title.toLowerCase().includes('looking')) {
          violations.gaze.push(violation);
        }
      } else if (alert.alert_type === 'object') {
        violations.object.push(violation);
      } else if (alert.alert_type === 'audio') {
        violations.audio.push(violation);
      }
    });
    
    // Call Python risk scoring engine
    const pythonScript = path.join(__dirname, '../utils/calculate_risk.py');
    const python = spawn('python', [
      pythonScript,
      JSON.stringify(violations),
      examDuration.toString()
    ]);
    
    let riskData = '';
    let errorData = '';
    
    python.stdout.on('data', (data) => {
      riskData += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    python.on('close', async (code) => {
      if (code !== 0) {
        logger.error(`Risk calculation error: ${errorData}`);
        // Fallback to simple calculation
        return res.json({
          session_id: sessionId,
          risk_score: calculateSimpleRiskScore(violations),
          violation_count: alerts.length,
          exam_duration_minutes: examDuration,
          violations: violations
        });
      }
      
      try {
        const riskScore = JSON.parse(riskData);
        
        // Store risk score in database
        await query(
          `INSERT INTO risk_scores 
           (session_id, risk_score, face_score, gaze_score, object_score, audio_score, duration_factor, severity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON DUPLICATE KEY UPDATE
           risk_score = VALUES(risk_score),
           face_score = VALUES(face_score),
           gaze_score = VALUES(gaze_score),
           object_score = VALUES(object_score),
           audio_score = VALUES(audio_score),
           duration_factor = VALUES(duration_factor),
           severity = VALUES(severity)`,
          [
            sessionId,
            riskScore.total_score,
            riskScore.breakdown.face_score,
            riskScore.breakdown.gaze_score,
            riskScore.breakdown.object_score,
            riskScore.breakdown.audio_score,
            riskScore.breakdown.duration_factor,
            riskScore.severity
          ]
        );
        
        res.json({
          session_id: sessionId,
          ...riskScore,
          violation_count: alerts.length,
          exam_duration_minutes: examDuration,
          violations: violations
        });
      } catch (e) {
        logger.error(`Risk score parse error: ${e.message}`);
        res.json({
          session_id: sessionId,
          risk_score: calculateSimpleRiskScore(violations),
          violation_count: alerts.length,
          exam_duration_minutes: examDuration,
          violations: violations
        });
      }
    });
    
  } catch (error) {
    logger.error('Risk score calculation error:', error);
    res.status(500).json({ error: 'Failed to calculate risk score' });
  }
});

/**
 * Get risk score history for a session
 */
router.get('/session/:sessionId/history', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const scores = await query(
      `SELECT * FROM risk_scores 
       WHERE session_id = ? 
       ORDER BY calculated_at DESC`,
      [sessionId]
    );
    
    res.json({ session_id: sessionId, scores });
  } catch (error) {
    logger.error('Risk score history error:', error);
    res.status(500).json({ error: 'Failed to fetch risk score history' });
  }
});

/**
 * Get risk scores for all sessions in an exam
 */
router.get('/exam/:examId/scores', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { examId } = req.params;
    
    const scores = await query(
      `SELECT rs.*, es.student_id, u.name as student_name
       FROM risk_scores rs
       JOIN exam_sessions es ON rs.session_id = es.id
       JOIN users u ON es.student_id = u.id
       WHERE es.exam_id = ?
       ORDER BY rs.risk_score DESC`,
      [examId]
    );
    
    res.json({ exam_id: examId, scores });
  } catch (error) {
    logger.error('Exam risk scores error:', error);
    res.status(500).json({ error: 'Failed to fetch exam risk scores' });
  }
});

/**
 * Simple fallback risk score calculation
 */
function calculateSimpleRiskScore(violations) {
  const weights = {
    face: 0.30,
    gaze: 0.25,
    object: 0.25,
    audio: 0.15
  };
  
  let score = 0;
  
  // Face violations
  if (violations.face.length > 0) {
    score += weights.face * 100;
  }
  
  // Gaze violations
  if (violations.gaze.length > 0) {
    score += weights.gaze * Math.min(100, violations.gaze.length * 20);
  }
  
  // Object violations
  if (violations.object.length > 0) {
    score += weights.object * 100;
  }
  
  // Audio violations
  if (violations.audio.length > 0) {
    score += weights.audio * Math.min(100, violations.audio.length * 30);
  }
  
  return Math.min(100, Math.round(score));
}

module.exports = router;
