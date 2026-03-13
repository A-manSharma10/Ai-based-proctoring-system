const express = require('express');
const Joi = require('joi');
const { query } = require('../config/database');
const { requireRole } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Get system stats
router.get('/stats', requireRole(['admin']), async (req, res) => {
    try {
        const userCount = await query('SELECT COUNT(*) as count FROM users');
        const examCount = await query('SELECT COUNT(*) as count FROM exams');
        const sessionCount = await query('SELECT COUNT(*) as count FROM exam_sessions');

        res.json({
            users: userCount[0].count,
            exams: examCount[0].count,
            activeSessions: sessionCount[0].count
        });
    } catch (error) {
        logger.error('Admin stats error:', error);
        res.status(500).json({ error: 'Failed to fetch admin stats' });
    }
});

// Create new exam
router.post('/exams', requireRole(['admin']), async (req, res) => {
    try {
        const { title, description, duration, start_time, end_time } = req.body;
        const result = await query(
            'INSERT INTO exams (title, description, duration, start_time, end_time) VALUES (?, ?, ?, ?, ?)',
            [title, description, duration, start_time || null, end_time || null]
        );
        res.status(201).json({ id: result.insertId, message: 'Exam created successfully' });
    } catch (error) {
        logger.error('Create exam error:', error);
        res.status(500).json({ error: 'Failed to create exam' });
    }
});

// Assign students to an exam session
router.post('/assign', requireRole(['admin']), async (req, res) => {
    try {
        const { examId, examName, studentIds, supervisorId, startTime } = req.body;

        for (const studentId of studentIds) {
            await query(
                'INSERT INTO exam_sessions (student_id, supervisor_id, exam_id, exam_name, start_time, status) VALUES (?, ?, ?, ?, ?, ?)',
                [studentId, supervisorId, examId, examName, startTime || new Date(), 'scheduled']
            );
        }
        res.status(201).json({ message: 'Students successfully assigned' });
    } catch (error) {
        logger.error('Assign students error:', error);
        res.status(500).json({ error: 'Failed to assign students' });
    }
});

// Add question to exam
router.post('/questions', requireRole(['admin', 'supervisor']), async (req, res) => {
    try {
        const { examName, questionNumber, questionText, questionType, options, points } = req.body;
        await query(
            'INSERT INTO exam_questions (exam_name, question_number, question_text, question_type, options, points) VALUES (?, ?, ?, ?, ?, ?)',
            [examName, questionNumber, questionText, questionType, JSON.stringify(options), points]
        );
        res.status(201).json({ message: 'Question added successfully' });
    } catch (error) {
        logger.error('Add question error:', error);
        res.status(500).json({ error: 'Failed to add question' });
    }
});

// Get all users
router.get('/users', requireRole(['admin']), async (req, res) => {
    try {
        const users = await query('SELECT id, name, email, role, created_at FROM users');
        res.json({ users });
    } catch (error) {
        logger.error('Get users error:', error);
        res.status(500).json({ error: 'Failed to fetch users' });
    }
});

// Get system health
router.get('/health', requireRole(['admin']), async (req, res) => {
    const AI_SERVICES = {
        face: process.env.FACE_SERVICE_URL || 'http://localhost:8000',
        object: process.env.OBJECT_SERVICE_URL || 'http://localhost:8001',
        audio: process.env.AUDIO_SERVICE_URL || 'http://localhost:8002',
        behavior: process.env.BEHAVIOR_SERVICE_URL || 'http://localhost:8003'
    };

    const health = {};
    const promises = Object.entries(AI_SERVICES).map(([name, url]) =>
        axios.get(`${url}/health`, { timeout: 2000 })
            .then(() => health[name] = 'healthy')
            .catch(() => health[name] = 'unreachable')
    );

    await Promise.all(promises);
    res.json(health);
});

// Get experiment metrics
router.get('/metrics', requireRole(['admin']), async (req, res) => {
    const fs = require('fs').promises;
    const path = require('path');
    const resultsDir = path.join(__dirname, '../../experiments/results');

    try {
        const [single, multi] = await Promise.all([
            fs.readFile(path.join(resultsDir, 'metrics_single_modal.json'), 'utf8')
                .then(JSON.parse).catch(() => null),
            fs.readFile(path.join(resultsDir, 'metrics_multimodal.json'), 'utf8')
                .then(JSON.parse).catch(() => null)
        ]);
        res.json({ single, multi });
    } catch (error) {
        res.status(500).json({ error: 'Failed to read metrics' });
    }
});

module.exports = router;
