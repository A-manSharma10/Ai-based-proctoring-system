const logger = require('../utils/logger');
const bcrypt = require('bcryptjs');

const mockData = {
    users: [
        { id: 1, email: 'student1@exam.com', password_hash: bcrypt.hashSync('password', 10), role: 'student', name: 'Student One', face_embedding: Buffer.from('mock_embedding') },
        { id: 2, email: 'student2@exam.com', password_hash: bcrypt.hashSync('password', 10), role: 'student', name: 'Student Two', face_embedding: Buffer.from('mock_embedding') },
        { id: 3, email: 'supervisor@exam.com', password_hash: bcrypt.hashSync('password', 10), role: 'supervisor', name: 'Supervisor One' },
        { id: 4, email: 'admin@exam.com', password_hash: bcrypt.hashSync('password', 10), role: 'admin', name: 'Admin User' }
    ],
    exams: [
        { id: 1, title: 'Sample Exam', description: 'This is a sample exam for demonstration.', duration: 60 }
    ],
    exam_questions: [
        { id: 1, exam_name: 'Sample Exam', question_number: 1, question_text: 'What is the capital of France?', question_type: 'multiple_choice', options: JSON.stringify(['Paris', 'London', 'Berlin', 'Madrid']), points: 1 },
        { id: 2, exam_name: 'Sample Exam', question_number: 2, question_text: 'What is 2 + 2?', question_type: 'multiple_choice', options: JSON.stringify(['3', '4', '5', '6']), points: 1 },
        { id: 3, exam_name: 'Sample Exam', question_number: 3, question_text: 'Which planet is closest to the Sun?', question_type: 'multiple_choice', options: JSON.stringify(['Venus', 'Mercury', 'Earth', 'Mars']), points: 1 },
        { id: 4, exam_name: 'Sample Exam', question_number: 4, question_text: 'Explain the concept of machine learning in your own words.', question_type: 'essay', options: '[]', points: 5 },
        { id: 5, exam_name: 'Sample Exam', question_number: 5, question_text: 'What does CPU stand for?', question_type: 'multiple_choice', options: JSON.stringify(['Central Processing Unit', 'Computer Power Unit', 'Core Programming Utility', 'Central Program Update']), points: 1 }
    ],
    exam_sessions: [
        { id: 1, student_id: 1, supervisor_id: 3, exam_name: 'Sample Exam', start_time: new Date(), end_time: null, status: 'active', student_name: 'Student One', student_email: 'student1@exam.com', supervisor_name: 'Supervisor One', supervisor_email: 'supervisor@exam.com', created_at: new Date() },
        { id: 2, student_id: 1, supervisor_id: 3, exam_name: 'Advanced AI Quiz', start_time: new Date(Date.now() + 86400000), end_time: null, status: 'scheduled', student_name: 'Student One', student_email: 'student1@exam.com', supervisor_name: 'Supervisor One', supervisor_email: 'supervisor@exam.com', created_at: new Date() },
        { id: 3, student_id: 1, supervisor_id: 3, exam_name: 'Security Protocol Test', start_time: new Date(Date.now() + 172800000), end_time: null, status: 'scheduled', student_name: 'Student One', student_email: 'student1@exam.com', supervisor_name: 'Supervisor One', supervisor_email: 'supervisor@exam.com', created_at: new Date() }
    ],
    student_answers: [],
    alerts: [],
    analysis_logs: []
};

async function initializeDatabase() {
    logger.info('Using MOCK Database for demonstration');
    return Promise.resolve();
}

async function query(sql, params = []) {
    logger.info(`Mock Query: ${sql.substring(0, 80)}`);

    // === USER QUERIES ===
    if (sql.includes('SELECT * FROM users WHERE email = ?')) {
        const user = mockData.users.find(u => u.email === params[0]);
        return user ? [user] : [];
    }
    if (sql.includes('SELECT id FROM users WHERE id = ?')) {
        const user = mockData.users.find(u => u.id === params[0]);
        return user ? [user] : [];
    }
    if (sql.includes('SELECT * FROM users')) {
        return mockData.users;
    }
    if (sql.includes('INSERT INTO users')) {
        const newUser = {
            id: mockData.users.length + 1,
            email: params[0],
            name: params[1],
            password_hash: params[2],
            role: params[3] || 'student'
        };
        mockData.users.push(newUser);
        return { insertId: newUser.id };
    }

    // === EXAM QUERIES ===
    if (sql.includes('FROM exams') || sql.includes('SELECT * FROM exams')) {
        return mockData.exams;
    }
    if (sql.includes('INSERT INTO exams')) {
        const newExam = { id: mockData.exams.length + 1, title: params[0], description: params[1], duration: params[2] };
        mockData.exams.push(newExam);
        return { insertId: newExam.id };
    }

    // === QUESTION QUERIES ===
    if (sql.includes('FROM exam_questions')) {
        if (sql.includes('COUNT(*)')) {
            const count = mockData.exam_questions.filter(q => !params[0] || q.exam_name === params[0]).length;
            return [{ count }];
        }
        const filtered = params[0] ? mockData.exam_questions.filter(q => q.exam_name === params[0]) : mockData.exam_questions;
        return filtered;
    }
    if (sql.includes('INSERT INTO exam_questions')) {
        const newQ = { id: mockData.exam_questions.length + 1, exam_name: params[0], question_number: params[1], question_text: params[2], question_type: params[3], options: params[4], points: params[5] };
        mockData.exam_questions.push(newQ);
        return { insertId: newQ.id };
    }

    // === SESSION QUERIES - CRITICAL: UPDATE must mutate state ===
    if (sql.includes('UPDATE exam_sessions SET status')) {
        // params layout: [newStatus, newTimestamp, sessionId]  or  [newStatus, sessionId]
        const newStatus = params[0];
        const sessionId = params[params.length - 1]; // always the last param
        const session = mockData.exam_sessions.find(s => s.id == sessionId);
        if (session) {
            session.status = newStatus;
            if (newStatus === 'active' && params[1] instanceof Date) session.start_time = params[1];
            if (newStatus === 'completed' && params[1] instanceof Date) session.end_time = params[1];
            logger.info(`Mock: Session ${sessionId} status -> '${newStatus}'`);
        }
        return { affectedRows: session ? 1 : 0 };
    }

    // === SESSION QUERIES - SELECT ===
    if (sql.includes('FROM exam_sessions')) {
        if (sql.includes('COUNT(*)')) return [{ count: mockData.exam_sessions.length }];

        // Complex detail query by session ID
        if (sql.includes('es.id = ?') || sql.includes('WHERE id = ?') || sql.includes('WHERE es.id = ?')) {
            const session = mockData.exam_sessions.find(s => s.id == params[0]);
            return session ? [session] : [];
        }
        // Filter by user
        if (params.length > 0) {
            return mockData.exam_sessions.filter(s => s.student_id == params[0] || s.supervisor_id == params[0]);
        }
        return mockData.exam_sessions;
    }

    // === SESSION - INSERT ===
    if (sql.includes('INSERT INTO exam_sessions')) {
        const newS = {
            id: mockData.exam_sessions.length + 1,
            student_id: params[0], supervisor_id: params[1],
            exam_name: params[2], start_time: params[3], status: params[4],
            end_time: null, created_at: new Date(),
            student_name: 'Student One', student_email: 'student1@exam.com',
            supervisor_name: 'Supervisor One', supervisor_email: 'supervisor@exam.com'
        };
        mockData.exam_sessions.push(newS);
        return { insertId: newS.id };
    }

    // === ANSWER QUERIES ===
    if (sql.includes('FROM student_answers')) {
        return mockData.student_answers.filter(a => a.session_id == params[0]);
    }
    if (sql.includes('INSERT INTO student_answers')) {
        const [sessionId, questionId, answer] = params;
        const existing = mockData.student_answers.find(a => a.session_id == sessionId && a.question_id == questionId);
        if (existing) {
            existing.answer = answer;
            existing.answered_at = new Date();
        } else {
            mockData.student_answers.push({ id: mockData.student_answers.length + 1, session_id: sessionId, question_id: questionId, answer, answered_at: new Date() });
        }
        return { affectedRows: 1 };
    }

    // === ALERT QUERIES ===
    if (sql.includes('INSERT INTO alerts')) {
        const newAlert = { id: mockData.alerts.length + 1, session_id: params[0], alert_type: params[1], severity: params[2], title: params[3], description: params[4], confidence_score: params[5], metadata: params[6], resolved: false, created_at: new Date().toISOString() };
        mockData.alerts.push(newAlert);
        return { insertId: newAlert.id };
    }
    if (sql.includes('FROM alerts')) {
        let result = [...mockData.alerts];

        // /session/:id route
        if (sql.includes('session_id = ?') || sql.includes('WHERE session_id = ?')) {
            result = result.filter(a => a.session_id == params[0]);
        }

        // /supervisor route — filter by sessions belonging to this supervisor
        if (sql.includes('es.supervisor_id = ?')) {
            const supervisorId = params[0];
            const supervisorSessions = mockData.exam_sessions.filter(s => s.supervisor_id == supervisorId);
            const sessionIds = supervisorSessions.map(s => s.id);
            result = result.filter(a => sessionIds.includes(a.session_id));
        }

        // Enrich with student + exam info
        result = result.map(alert => {
            const session = mockData.exam_sessions.find(s => s.id == alert.session_id);
            const student = session ? mockData.users.find(u => u.id == session.student_id) : null;
            return {
                ...alert,
                exam_name: session?.exam_name || 'Unknown',
                student_name: student?.name || 'Unknown Student',
                metadata: alert.metadata
            };
        });

        // Apply resolved filter if present
        if (sql.includes('a.resolved = ?')) {
            const resolvedParam = params.find(p => typeof p === 'boolean');
            if (resolvedParam !== undefined) result = result.filter(a => a.resolved === resolvedParam);
        }

        return result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)).slice(0, 100);
    }
    if (sql.includes('UPDATE alerts SET resolved')) {
        const alertId = params[params.length - 1];
        const alert = mockData.alerts.find(a => a.id == alertId);
        if (alert) {
            alert.resolved = params[0];
            alert.resolved_at = params[2];
        }
        return { affectedRows: alert ? 1 : 0 };
    }

    // === ANALYSIS LOG QUERIES ===
    if (sql.includes('INSERT INTO analysis_logs')) {
        mockData.analysis_logs.push({ id: mockData.analysis_logs.length + 1, session_id: params[0], service_type: params[1], analysis_data: params[2], processing_time_ms: params[3], frame_timestamp: params[4] });
        return { insertId: mockData.analysis_logs.length };
    }
    if (sql.includes('FROM analysis_logs')) {
        return mockData.analysis_logs.filter(l => l.session_id == params[0]);
    }

    // Generic fallback
    const sqlUpper = sql.trim().toUpperCase();
    if (sql.includes('SELECT COUNT(*)')) return [{ count: 5 }];
    if (sqlUpper.startsWith('UPDATE') || sqlUpper.startsWith('DELETE')) return { affectedRows: 1 };

    return [];
}

async function transaction(callback) {
    logger.info('Mock Transaction started');
    return callback({ execute: query });
}

module.exports = {
    initializeDatabase,
    query,
    transaction,
    pool: { execute: query },
    mockData
};
