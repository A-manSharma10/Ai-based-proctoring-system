const express = require('express');
const Joi = require('joi');
const { query, transaction } = require('../config/database');
const { requireRole } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const createSessionSchema = Joi.object({
  studentId: Joi.number().integer().positive().required(),
  examName: Joi.string().min(1).max(255).required(),
  startTime: Joi.date().iso().optional()
});

const submitAnswerSchema = Joi.object({
  questionId: Joi.number().integer().positive().required(),
  answer: Joi.string().required()
});

// Get exam questions
router.get('/questions/:examName', async (req, res) => {
  try {
    const { examName } = req.params;

    const questions = await query(
      'SELECT id, question_number, question_text, question_type, options, points FROM exam_questions WHERE exam_name = ? ORDER BY question_number',
      [examName]
    );

    res.json({ questions });
  } catch (error) {
    logger.error('Get questions error:', error);
    res.status(500).json({ error: 'Failed to fetch questions' });
  }
});

// Create exam session
router.post('/session', requireRole(['supervisor', 'admin']), async (req, res) => {
  try {
    const { error, value } = createSessionSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { studentId, examName, startTime } = value;
    const supervisorId = req.user.userId;

    // Check if student exists
    const student = await query('SELECT id FROM users WHERE id = ? AND role = "student"', [studentId]);
    if (student.length === 0) {
      return res.status(404).json({ error: 'Student not found' });
    }

    // Check if exam questions exist
    const questions = await query('SELECT COUNT(*) as count FROM exam_questions WHERE exam_name = ?', [examName]);
    if (questions[0].count === 0) {
      return res.status(404).json({ error: 'Exam not found' });
    }

    // Create session
    const result = await query(
      'INSERT INTO exam_sessions (student_id, supervisor_id, exam_name, start_time, status) VALUES (?, ?, ?, ?, ?)',
      [studentId, supervisorId, examName, startTime || new Date(), 'scheduled']
    );

    res.status(201).json({
      sessionId: result.insertId,
      message: 'Exam session created successfully'
    });
  } catch (error) {
    logger.error('Create session error:', error);
    res.status(500).json({ error: 'Failed to create exam session' });
  }
});

// Start exam session
router.post('/session/:sessionId/start', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.user.userId;

    // Get session details
    const sessions = await query(
      'SELECT * FROM exam_sessions WHERE id = ? AND (student_id = ? OR supervisor_id = ?)',
      [sessionId, userId, userId]
    );

    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const session = sessions[0];

    // Allow re-joining active sessions or starting scheduled ones
    if (session.status !== 'scheduled' && session.status !== 'active') {
      return res.status(400).json({ error: 'Session cannot be started' });
    }

    // Update session status
    await query(
      'UPDATE exam_sessions SET status = ?, start_time = ? WHERE id = ?',
      ['active', new Date(), sessionId]
    );

    res.json({ message: 'Exam session started successfully' });
  } catch (error) {
    logger.error('Start session error:', error);
    res.status(500).json({ error: 'Failed to start exam session' });
  }
});

// End exam session
router.post('/session/:sessionId/end', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.user.userId;

    // Get session details
    const sessions = await query(
      'SELECT * FROM exam_sessions WHERE id = ? AND (student_id = ? OR supervisor_id = ?)',
      [sessionId, userId, userId]
    );

    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const session = sessions[0];

    if (session.status !== 'active' && session.status !== 'scheduled') {
      return res.status(400).json({ error: 'Session is not active or available' });
    }

    // Update session status
    await query(
      'UPDATE exam_sessions SET status = ?, end_time = ? WHERE id = ?',
      ['completed', new Date(), sessionId]
    );

    res.json({ message: 'Exam session ended successfully' });
  } catch (error) {
    logger.error('End session error:', error);
    res.status(500).json({ error: 'Failed to end exam session' });
  }
});

// Submit answer
router.post('/session/:sessionId/answer', requireRole(['student']), async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { error, value } = submitAnswerSchema.validate(req.body);

    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { questionId, answer } = value;

    // Insert or update answer - simplified: just store without strict session status check
    await query(
      'INSERT INTO student_answers (session_id, question_id, answer) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE answer = ?, answered_at = CURRENT_TIMESTAMP',
      [sessionId, questionId, answer, answer]
    );

    res.json({ message: 'Answer submitted successfully' });
  } catch (error) {
    logger.error('Submit answer error:', error);
    res.status(500).json({ error: 'Failed to submit answer' });
  }
});

// Get session details
router.get('/session/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.user.userId;

    // Get session with user details
    const sessions = await query(`
      SELECT 
        es.*,
        u1.name as student_name,
        u1.email as student_email,
        u2.name as supervisor_name,
        u2.email as supervisor_email
      FROM exam_sessions es
      LEFT JOIN users u1 ON es.student_id = u1.id
      LEFT JOIN users u2 ON es.supervisor_id = u2.id
      WHERE es.id = ? AND (es.student_id = ? OR es.supervisor_id = ? OR ? IN (SELECT id FROM users WHERE role = 'admin'))
    `, [sessionId, userId, userId, userId]);

    if (sessions.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const session = sessions[0];

    // Get answers if student
    let answers = [];
    if (req.user.role === 'student' && session.student_id === userId) {
      answers = await query(
        'SELECT question_id, answer, answered_at FROM student_answers WHERE session_id = ?',
        [sessionId]
      );
    }

    res.json({ session, answers });
  } catch (error) {
    logger.error('Get session error:', error);
    res.status(500).json({ error: 'Failed to fetch session details' });
  }
});

// Get user sessions
router.get('/sessions', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { role } = req.user;

    let query_str, params;

    if (role === 'student') {
      query_str = `
        SELECT 
          es.*,
          u2.name as supervisor_name
        FROM exam_sessions es
        LEFT JOIN users u2 ON es.supervisor_id = u2.id
        WHERE es.student_id = ?
        ORDER BY es.created_at DESC
      `;
      params = [userId];
    } else if (role === 'supervisor') {
      query_str = `
        SELECT 
          es.*,
          u1.name as student_name,
          u1.email as student_email
        FROM exam_sessions es
        LEFT JOIN users u1 ON es.student_id = u1.id
        WHERE es.supervisor_id = ?
        ORDER BY es.created_at DESC
      `;
      params = [userId];
    } else {
      // Admin can see all sessions
      query_str = `
        SELECT 
          es.*,
          u1.name as student_name,
          u1.email as student_email,
          u2.name as supervisor_name
        FROM exam_sessions es
        LEFT JOIN users u1 ON es.student_id = u1.id
        LEFT JOIN users u2 ON es.supervisor_id = u2.id
        ORDER BY es.created_at DESC
      `;
      params = [];
    }

    const sessions = await query(query_str, params);
    res.json({ sessions });
  } catch (error) {
    logger.error('Get sessions error:', error);
    res.status(500).json({ error: 'Failed to fetch sessions' });
  }
});

module.exports = router;