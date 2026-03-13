const express = require('express');
const Joi = require('joi');
const axios = require('axios');
const { query } = require('../config/database');
const { requireRole } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// AI Service URLs
const AI_SERVICES = {
  face: process.env.FACE_SERVICE_URL || 'http://face_service:8000',
  object: process.env.OBJECT_SERVICE_URL || 'http://object_service:8000',
  audio: process.env.AUDIO_SERVICE_URL || 'http://audio_service:8000',
  behavior: process.env.BEHAVIOR_SERVICE_URL || 'http://behavior_service:8000'
};

// Validation schemas
const analyzeFrameSchema = Joi.object({
  sessionId: Joi.number().integer().positive().required(),
  frameData: Joi.string().required(), // base64 encoded image
  timestamp: Joi.date().iso().optional(),
  services: Joi.array().items(Joi.string().valid('face', 'object', 'behavior')).default(['face', 'object', 'behavior'])
});

const analyzeAudioSchema = Joi.object({
  sessionId: Joi.number().integer().positive().required(),
  audioData: Joi.string().required(), // base64 encoded audio
  timestamp: Joi.date().iso().optional()
});

// Analyze video frame
router.post('/frame', async (req, res) => {
  try {
    const { error, value } = analyzeFrameSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { sessionId, frameData, timestamp, services } = value;
    const frameTimestamp = timestamp ? new Date(timestamp) : new Date();

    // Verify session exists and is active
    const sessions = await query(
      'SELECT * FROM exam_sessions WHERE id = ? AND status = "active"',
      [sessionId]
    );

    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Active session not found' });
    }

    const analysisResults = {};
    const analysisPromises = [];

    if (process.env.USE_MOCK === 'true') {
      logger.info('Using PROFESSIONAL MOCK AI Analysis');

      // Default: No violations
      if (services.includes('face')) {
        // Mock: 95% chance of face being present
        const facePresent = Math.random() > 0.05;
        analysisResults.face = {
          face_detected: facePresent,
          face_count: facePresent ? 1 : 0,
          confidence: 0.99,
          attention_score: facePresent ? (Math.random() > 0.1 ? 0.9 : 0.2) : 0
        };
      }
      if (services.includes('object')) analysisResults.object = { detections: [], prohibited_objects: [] };
      if (services.includes('behavior')) analysisResults.behavior = { risk_score: 0.05, detected_behaviors: [] };

      // Professional Mocking: Inject realistic violations 8% of the time
      if (Math.random() > 0.92) {
        const rand = Math.random();
        if (rand > 0.75) {
          // Face Violation
          analysisResults.face.violation = {
            type: 'no_face',
            message: 'Student has left the frame',
            severity: 'critical',
            confidence: 0.95,
            duration: 2.5
          };
        } else if (rand > 0.5) {
          // Object Violation
          analysisResults.object.violation = {
            object: 'cell phone',
            message: 'Suspicious device detected',
            severity: 'critical',
            confidence: 0.88,
            duration: 1.2
          };
        } else if (rand > 0.25) {
          // Behavioral Violation
          analysisResults.behavior.detected_behaviors = [{
            pattern_type: 'unauthorized_hands',
            message: 'Multiple hands detected in frame',
            severity: 'high',
            confidence: 0.91
          }];
          analysisResults.behavior.risk_score = 0.75;
        } else {
          // Gaze Violation
          analysisResults.face.violation = {
            type: 'looking_away',
            message: 'Extended gaze away from screen detected',
            severity: 'medium',
            confidence: 0.82,
            duration: 5.0
          };
        }
      }
    } else {
      // Face analysis
      if (services.includes('face')) {
        analysisPromises.push(
          axios.post(`${AI_SERVICES.face}/analyze`, {
            image: frameData,
            session_id: sessionId
          }).then(response => {
            analysisResults.face = response.data;
          }).catch(error => {
            logger.error('Face analysis error:', error.message);
            analysisResults.face = { error: 'Face analysis failed' };
          })
        );
      }

      // Object detection
      if (services.includes('object')) {
        analysisPromises.push(
          axios.post(`${AI_SERVICES.object}/detect`, {
            image: frameData,
            session_id: sessionId
          }).then(response => {
            analysisResults.object = response.data;
          }).catch(error => {
            logger.error('Object detection error:', error.message);
            analysisResults.object = { error: 'Object detection failed' };
          })
        );
      }

      // Behavioral analysis
      if (services.includes('behavior')) {
        analysisPromises.push(
          axios.post(`${AI_SERVICES.behavior}/analyze`, {
            image: frameData,
            session_id: sessionId
          }).then(response => {
            analysisResults.behavior = response.data;
          }).catch(error => {
            logger.error('Behavioral analysis error:', error.message);
            analysisResults.behavior = { error: 'Behavioral analysis failed' };
          })
        );
      }

      await Promise.all(analysisPromises);
    }

    // Store analysis results
    const logPromises = [];
    for (const [serviceType, result] of Object.entries(analysisResults)) {
      if (!result.error) {
        logPromises.push(
          query(
            'INSERT INTO analysis_logs (session_id, service_type, analysis_data, processing_time_ms, frame_timestamp) VALUES (?, ?, ?, ?, ?)',
            [sessionId, serviceType, JSON.stringify(result), result.processing_time || 0, frameTimestamp]
          )
        );
      }
    }

    await Promise.all(logPromises);

    // Generate alerts based on analysis results and include frame evidence
    await generateAlerts(sessionId, analysisResults, frameData);

    res.json({
      sessionId,
      timestamp: frameTimestamp,
      results: analysisResults
    });
  } catch (error) {
    logger.error('Frame analysis error:', error);
    res.status(500).json({ error: 'Frame analysis failed' });
  }
});

// Analyze audio
router.post('/audio', async (req, res) => {
  try {
    const { error, value } = analyzeAudioSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { sessionId, audioData, timestamp } = value;
    const audioTimestamp = timestamp ? new Date(timestamp) : new Date();

    // Verify session exists and is active
    const sessions = await query(
      'SELECT * FROM exam_sessions WHERE id = ? AND status = "active"',
      [sessionId]
    );

    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Active session not found' });
    }

    // Audio analysis
    let analysisResult;
    if (process.env.USE_MOCK === 'true') {
      logger.info('Using PROFESSIONAL MOCK Audio Analysis');
      analysisResult = { is_whisper: false, speaker_count: 1, confidence: 0.98 };

      // Inject professional audio violations 10% of the time
      if (Math.random() > 0.9) {
        analysisResult.violation = {
          type: 'multiple_speakers',
          message: 'Detected multiple distinct voices in the environment',
          severity: 'high',
          confidence: 0.94,
          speaker_count: 2
        };
      }
    } else {
      try {
        const response = await axios.post(`${AI_SERVICES.audio}/analyze`, {
          audio: audioData,
          session_id: sessionId
        });
        analysisResult = response.data;
      } catch (error) {
        logger.error('Audio analysis error:', error.message);
        return res.status(500).json({ error: 'Audio analysis failed' });
      }
    }

    // Store analysis result
    await query(
      'INSERT INTO analysis_logs (session_id, service_type, analysis_data, processing_time_ms, frame_timestamp) VALUES (?, ?, ?, ?, ?)',
      [sessionId, 'audio', JSON.stringify(analysisResult), analysisResult.processing_time || 0, audioTimestamp]
    );

    // Generate alerts based on audio analysis
    await generateAudioAlerts(sessionId, analysisResult);

    res.json({
      sessionId,
      timestamp: audioTimestamp,
      result: analysisResult
    });
  } catch (error) {
    logger.error('Audio analysis error:', error);
    res.status(500).json({ error: 'Audio analysis failed' });
  }
});

// Get analysis history
router.get('/history/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { serviceType, limit = 100, offset = 0 } = req.query;

    let whereClause = 'WHERE session_id = ?';
    let params = [sessionId];

    if (serviceType) {
      whereClause += ' AND service_type = ?';
      params.push(serviceType);
    }

    const logs = await query(`
      SELECT *
      FROM analysis_logs
      ${whereClause}
      ORDER BY frame_timestamp DESC
      LIMIT ? OFFSET ?
    `, [...params, parseInt(limit), parseInt(offset)]);

    // Parse analysis data JSON
    const logsWithData = logs.map(log => ({
      ...log,
      analysis_data: log.analysis_data ? JSON.parse(log.analysis_data) : null
    }));

    res.json({ logs: logsWithData });
  } catch (error) {
    logger.error('Get analysis history error:', error);
    res.status(500).json({ error: 'Failed to fetch analysis history' });
  }
});

// Get analysis summary
router.get('/summary/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;

    // Get analysis counts by service type
    const summary = await query(`
      SELECT 
        service_type,
        COUNT(*) as total_analyses,
        AVG(processing_time_ms) as avg_processing_time,
        MIN(frame_timestamp) as first_analysis,
        MAX(frame_timestamp) as last_analysis
      FROM analysis_logs
      WHERE session_id = ?
      GROUP BY service_type
    `, [sessionId]);

    // Get alert summary
    const alertSummary = await query(`
      SELECT 
        alert_type,
        severity,
        COUNT(*) as count
      FROM alerts
      WHERE session_id = ?
      GROUP BY alert_type, severity
    `, [sessionId]);

    res.json({
      sessionId,
      analysisSummary: summary,
      alertSummary
    });
  } catch (error) {
    logger.error('Get analysis summary error:', error);
    res.status(500).json({ error: 'Failed to fetch analysis summary' });
  }
});

// In-memory state for temporal smoothing and risk tracking
const sessionStates = {};

function getSessionState(sessionId) {
  if (!sessionStates[sessionId]) {
    sessionStates[sessionId] = {
      // Face temporal counters
      noFaceFrames: 0,
      lookingAwayFrames: 0,
      multipleFacesFrames: 0,
      // Object temporal counters
      phoneFrames: 0,
      // Thresholds (number of consecutive frames, e.g. at 1 frame per 2s)
      thresholds: {
        noFace: 2,       // ~4s
        lookingAway: 2,  // ~4s
        multipleFaces: 1 // immediate
      },
      // Overall
      riskScore: 0,
      startTime: Date.now(),
      violationsCount: 0,
      maxViolations: 5,
      isTerminated: false
    };
  }
  return sessionStates[sessionId];
}

// Calculate explainable Risk Score
function calculateRiskScore(state, analysisResults) {
  // Weights for different violation types
  const weights = {
    faceAbsent: 0.3,
    multipleFaces: 0.35,
    lookingAway: 0.15,
    prohibitedObject: 0.4,
    suspiciousAudio: 0.2,
    durationFactor: 0.05
  };

  let risk = 0;

  // Base risk from historical violations (slow decay can be added, but here we incrementally boost)
  risk += Math.min(0.4, state.violationsCount * 0.05);

  // Add immediate risks from current frame
  if (analysisResults.face) {
    if (!analysisResults.face.face_detected) risk += weights.faceAbsent;
    if (analysisResults.face.multiple_faces) risk += weights.multipleFaces;
    if (analysisResults.face.attention_score < 0.4) risk += weights.lookingAway;
  }

  if (analysisResults.object && analysisResults.object.prohibited_objects?.length > 0) {
    risk += weights.prohibitedObject;
  }

  // Duration factor (longer exam = slightly higher baseline risk tolerance, but here we keep it simple)
  // Let's cap risk at 1.0 (100%)

  return Math.min(1.0, risk);
}

// Helper function to generate alerts based on professional AI service violations
async function generateAlerts(sessionId, analysisResults, frameData = null) {
  const alerts = [];
  const state = getSessionState(sessionId);

  if (state.isTerminated) return;

  // 1. Process Temporal Smoothing for Face
  if (analysisResults.face) {
    // Face Detection smoothing
    if (!analysisResults.face.face_detected) {
      state.noFaceFrames++;
      if (state.noFaceFrames >= state.thresholds.noFace) {
        alerts.push({
          alertType: 'face',
          severity: 'critical',
          title: 'FACE NOT DETECTED',
          description: 'No face detected in the frame for a prolonged period.',
          confidenceScore: 0.99
        });
        state.noFaceFrames = 0; // reset after alert
      }
    } else {
      state.noFaceFrames = 0;
    }

    // Gaze Detection smoothing
    if (analysisResults.face.face_detected && analysisResults.face.attention_score < 0.3) {
      state.lookingAwayFrames++;
      if (state.lookingAwayFrames >= state.thresholds.lookingAway) {
        alerts.push({
          alertType: 'face',
          severity: 'medium',
          title: 'GAZE DEVIATION',
          description: 'Attention diverted away from screen.',
          confidenceScore: 0.85
        });
        state.lookingAwayFrames = 0;
      }
    } else {
      state.lookingAwayFrames = 0;
    }

    // Multiple Persons
    if (analysisResults.face.face_count > 1) {
      alerts.push({
        alertType: 'face',
        severity: 'critical',
        title: 'MULTIPLE PERSONS',
        description: 'Multiple faces detected in the field of view.',
        confidenceScore: 0.95
      });
    }
  }

  // 2. Prohibited Objects (Immediate or smoothed)
  if (analysisResults.object && analysisResults.object.prohibited_objects?.length > 0) {
    analysisResults.object.prohibited_objects.forEach(obj => {
      alerts.push({
        alertType: 'object',
        severity: 'critical',
        title: `PROHIBITED OBJECT: ${obj.toUpperCase()}`,
        description: `Detection of unauthorized device: ${obj}`,
        confidenceScore: 0.9
      });
    });
  }

  // Inject metadata and evidence
  alerts.forEach(alert => {
    alert.sessionId = sessionId;
    alert.metadata = {
      evidence: frameData,
      riskScoreAtEvent: state.riskScore
    };
  });

  // Record and emit professionally tracked alerts
  for (const alert of alerts) {
    state.violationsCount += 1;

    // Check for Termination
    if (state.violationsCount >= state.maxViolations && !state.isTerminated) {
      state.isTerminated = true;
      await terminateSession(sessionId, 'Exceeded maximum violation threshold');
    }

    try {
      await query(
        'INSERT INTO alerts (session_id, alert_type, severity, title, description, confidence_score, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [alert.sessionId, alert.alertType, alert.severity, alert.title, alert.description, alert.confidenceScore, JSON.stringify(alert.metadata)]
      );

      const { io } = require('../server');
      if (io) {
        io.to(`session_${sessionId}`).emit('violation-warning', alert);
        io.to(`supervisor_${sessionId}`).emit('new-alert', alert);
        io.emit('new-alert', alert);
      }
    } catch (error) {
      logger.error('Failed to register alert:', error);
    }
  }
}

async function terminateSession(sessionId, reason) {
  logger.warn(`TERMINATING SESSION ${sessionId}: ${reason}`);
  try {
    await query(
      'UPDATE exam_sessions SET status = "terminated", end_time = CURRENT_TIMESTAMP, metadata = JSON_SET(COALESCE(metadata, "{}"), "$.terminationReason", ?) WHERE id = ?',
      [reason, sessionId]
    );

    const { io } = require('../server');
    if (io) {
      io.to(`session_${sessionId}`).emit('session-terminated', { reason });
    }
  } catch (error) {
    logger.error('Failed to terminate session:', error);
  }
}

// Helper function to generate professional audio alerts
async function generateAudioAlerts(sessionId, analysisResult) {
  if (!analysisResult.violation) return;

  const state = getSessionState(sessionId);
  if (state.isTerminated) return;

  const v = analysisResult.violation;
  const alert = {
    sessionId,
    alertType: 'audio',
    severity: v.severity || 'medium',
    title: v.type.replace(/_/g, ' ').toUpperCase(),
    description: v.message,
    confidenceScore: v.confidence,
    metadata: { ...v, is_whisper: analysisResult.is_whisper }
  };

  state.violationsCount += 1;
  if (state.violationsCount >= state.maxViolations && !state.isTerminated) {
    state.isTerminated = true;
    await terminateSession(sessionId, 'Exceeded maximum violation threshold (Audio/Other)');
  }

  try {
    await query(
      'INSERT INTO alerts (session_id, alert_type, severity, title, description, confidence_score, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [sessionId, alert.alertType, alert.severity, alert.title, alert.description, alert.confidenceScore, JSON.stringify(alert.metadata)]
    );

    const { io } = require('../server');
    if (io) {
      io.to(`session_${sessionId}`).emit('violation-warning', alert);
      io.to(`supervisor_${sessionId}`).emit('new-alert', alert);
      io.emit('new-alert', alert);
    }
  } catch (error) {
    logger.error('Failed to create audio alert:', error);
  }
}


module.exports = router;