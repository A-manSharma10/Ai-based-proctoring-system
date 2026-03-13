const express = require('express');
const Joi = require('joi');
const { query } = require('../config/database');
const { requireRole } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const createAlertSchema = Joi.object({
  sessionId: Joi.number().integer().positive().required(),
  alertType: Joi.string().valid('face', 'object', 'audio', 'behavior', 'system').required(),
  severity: Joi.string().valid('low', 'medium', 'high', 'critical').required(),
  title: Joi.string().min(1).max(255).required(),
  description: Joi.string().required(),
  confidenceScore: Joi.number().min(0).max(1).optional(),
  metadata: Joi.object().optional()
});

const resolveAlertSchema = Joi.object({
  resolved: Joi.boolean().required(),
  notes: Joi.string().optional()
});

// Create alert
router.post('/', async (req, res) => {
  try {
    const { error, value } = createAlertSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { sessionId, alertType, severity, title, description, confidenceScore, metadata } = value;

    // Verify session exists
    const sessions = await query('SELECT id FROM exam_sessions WHERE id = ?', [sessionId]);
    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }

    // Create alert
    const result = await query(
      'INSERT INTO alerts (session_id, alert_type, severity, title, description, confidence_score, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [sessionId, alertType, severity, title, description, confidenceScore, JSON.stringify(metadata)]
    );

    const alert = {
      id: result.insertId,
      sessionId,
      alertType,
      severity,
      title,
      description,
      confidenceScore,
      metadata,
      resolved: false,
      createdAt: new Date()
    };

    // Emit alert to connected supervisors via Socket.IO
    const { io } = require('../server');
    if (io) {
      io.emit('new-alert', alert);
    }

    res.status(201).json({ alert, message: 'Alert created successfully' });
  } catch (error) {
    logger.error('Create alert error:', error);
    res.status(500).json({ error: 'Failed to create alert' });
  }
});

// Get alerts for session
router.get('/session/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { resolved, alertType, severity } = req.query;

    let whereClause = 'WHERE session_id = ?';
    let params = [sessionId];

    if (resolved !== undefined) {
      whereClause += ' AND resolved = ?';
      params.push(resolved === 'true');
    }

    if (alertType) {
      whereClause += ' AND alert_type = ?';
      params.push(alertType);
    }

    if (severity) {
      whereClause += ' AND severity = ?';
      params.push(severity);
    }

    const alerts = await query(`
      SELECT 
        a.*,
        u.name as resolved_by_name
      FROM alerts a
      LEFT JOIN users u ON a.resolved_by = u.id
      ${whereClause}
      ORDER BY a.created_at DESC
    `, params);

    // Parse metadata JSON
    const alertsWithMetadata = alerts.map(alert => ({
      ...alert,
      metadata: alert.metadata ? JSON.parse(alert.metadata) : null
    }));

    res.json({ alerts: alertsWithMetadata });
  } catch (error) {
    logger.error('Get alerts error:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

// Get alerts for supervisor
router.get('/supervisor', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const supervisorId = req.user.userId;
    const { resolved, alertType, severity, limit = 50 } = req.query;

    let whereClause = 'WHERE es.supervisor_id = ?';
    let params = [supervisorId];

    if (resolved !== undefined) {
      whereClause += ' AND a.resolved = ?';
      params.push(resolved === 'true');
    }

    if (alertType) {
      whereClause += ' AND a.alert_type = ?';
      params.push(alertType);
    }

    if (severity) {
      whereClause += ' AND a.severity = ?';
      params.push(severity);
    }

    const alerts = await query(`
      SELECT 
        a.*,
        es.exam_name,
        u1.name as student_name,
        u2.name as resolved_by_name
      FROM alerts a
      JOIN exam_sessions es ON a.session_id = es.id
      JOIN users u1 ON es.student_id = u1.id
      LEFT JOIN users u2 ON a.resolved_by = u2.id
      ${whereClause}
      ORDER BY a.created_at DESC
      LIMIT ?
    `, [...params, parseInt(limit)]);

    // Parse metadata JSON
    const alertsWithMetadata = alerts.map(alert => ({
      ...alert,
      metadata: alert.metadata ? JSON.parse(alert.metadata) : null
    }));

    res.json({ alerts: alertsWithMetadata });
  } catch (error) {
    logger.error('Get supervisor alerts error:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

// Resolve alert
router.patch('/:alertId/resolve', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { alertId } = req.params;
    const { error, value } = resolveAlertSchema.validate(req.body);
    
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { resolved, notes } = value;
    const resolvedBy = req.user.userId;

    // Update alert
    const result = await query(
      'UPDATE alerts SET resolved = ?, resolved_by = ?, resolved_at = ? WHERE id = ?',
      [resolved, resolvedBy, resolved ? new Date() : null, alertId]
    );

    if (result.affectedRows === 0) {
      return res.status(404).json({ error: 'Alert not found' });
    }

    // Log resolution if notes provided
    if (notes) {
      await query(
        'INSERT INTO analysis_logs (session_id, service_type, analysis_data) SELECT session_id, "system", ? FROM alerts WHERE id = ?',
        [JSON.stringify({ action: 'alert_resolved', notes, resolvedBy }), alertId]
      );
    }

    res.json({ message: 'Alert updated successfully' });
  } catch (error) {
    logger.error('Resolve alert error:', error);
    res.status(500).json({ error: 'Failed to resolve alert' });
  }
});

// Get alert statistics
router.get('/stats', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { sessionId, timeRange = '24h' } = req.query;
    
    let timeCondition = '';
    let params = [];

    // Add time range condition
    if (timeRange === '1h') {
      timeCondition = 'AND a.created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)';
    } else if (timeRange === '24h') {
      timeCondition = 'AND a.created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)';
    } else if (timeRange === '7d') {
      timeCondition = 'AND a.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)';
    }

    let sessionCondition = '';
    if (sessionId) {
      sessionCondition = 'AND a.session_id = ?';
      params.push(sessionId);
    }

    // Get alert counts by type and severity
    const stats = await query(`
      SELECT 
        a.alert_type,
        a.severity,
        COUNT(*) as count,
        AVG(a.confidence_score) as avg_confidence,
        SUM(CASE WHEN a.resolved = 1 THEN 1 ELSE 0 END) as resolved_count
      FROM alerts a
      WHERE 1=1 ${timeCondition} ${sessionCondition}
      GROUP BY a.alert_type, a.severity
      ORDER BY count DESC
    `, params);

    // Get total counts
    const totals = await query(`
      SELECT 
        COUNT(*) as total_alerts,
        SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved_alerts,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
        SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_alerts
      FROM alerts a
      WHERE 1=1 ${timeCondition} ${sessionCondition}
    `, params);

    res.json({
      stats,
      totals: totals[0] || {
        total_alerts: 0,
        resolved_alerts: 0,
        critical_alerts: 0,
        high_alerts: 0
      }
    });
  } catch (error) {
    logger.error('Get alert stats error:', error);
    res.status(500).json({ error: 'Failed to fetch alert statistics' });
  }
});

module.exports = router;