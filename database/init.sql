-- AI Exam Proctoring System Database Schema

CREATE DATABASE IF NOT EXISTS exam_proctoring;
USE exam_proctoring;

-- Users table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('student', 'supervisor', 'admin') NOT NULL DEFAULT 'student',
    face_embedding LONGBLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_role (role)
);

-- Exams table (Blueprint)
CREATE TABLE exams (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    duration INT NOT NULL,  -- in minutes
    start_time TIMESTAMP NULL,
    end_time TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Exam sessions table
CREATE TABLE exam_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    supervisor_id INT,
    exam_id INT NULL,
    exam_name VARCHAR(255) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NULL,
    status ENUM('scheduled', 'active', 'completed', 'terminated') NOT NULL DEFAULT 'scheduled',
    recording_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (supervisor_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE,
    INDEX idx_student_id (student_id),
    INDEX idx_supervisor_id (supervisor_id),
    INDEX idx_exam_id (exam_id),
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
);

-- Alerts table
CREATE TABLE alerts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    alert_type ENUM('face', 'object', 'audio', 'behavior', 'system') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    confidence_score DECIMAL(5,4),
    metadata JSON,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by INT NULL,
    resolved_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (resolved_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_session_id (session_id),
    INDEX idx_alert_type (alert_type),
    INDEX idx_severity (severity),
    INDEX idx_resolved (resolved),
    INDEX idx_created_at (created_at)
);

-- Analysis logs table
CREATE TABLE analysis_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    service_type ENUM('face', 'object', 'audio', 'behavior') NOT NULL,
    analysis_data JSON,
    processing_time_ms INT,
    frame_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_service_type (service_type),
    INDEX idx_frame_timestamp (frame_timestamp)
);

-- Exam questions table
CREATE TABLE exam_questions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    exam_name VARCHAR(255) NOT NULL,
    question_number INT NOT NULL,
    question_text TEXT NOT NULL,
    question_type ENUM('multiple_choice', 'text', 'essay') NOT NULL,
    options JSON,
    correct_answer TEXT,
    points INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_exam_name (exam_name),
    INDEX idx_question_number (question_number)
);

-- Student answers table
CREATE TABLE student_answers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    question_id INT NOT NULL,
    answer TEXT,
    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES exam_questions(id) ON DELETE CASCADE,
    UNIQUE KEY unique_session_question (session_id, question_id),
    INDEX idx_session_id (session_id),
    INDEX idx_question_id (question_id)
);

-- Insert sample data
INSERT INTO users (email, name, password_hash, role) VALUES
('admin@exam.com', 'System Admin', '$2b$10$rQZ8kHp0rQZ8kHp0rQZ8kOp0rQZ8kHp0rQZ8kHp0rQZ8kHp0rQZ8k', 'admin'),
('supervisor@exam.com', 'John Supervisor', '$2b$10$rQZ8kHp0rQZ8kHp0rQZ8kOp0rQZ8kHp0rQZ8kHp0rQZ8kHp0rQZ8k', 'supervisor'),
('student1@exam.com', 'Alice Student', '$2b$10$rQZ8kHp0rQZ8kHp0rQZ8kOp0rQZ8kHp0rQZ8kHp0rQZ8kHp0rQZ8k', 'student'),
('student2@exam.com', 'Bob Student', '$2b$10$rQZ8kHp0rQZ8kHp0rQZ8kOp0rQZ8kHp0rQZ8kHp0rQZ8kHp0rQZ8k', 'student');

-- Insert sample exams
INSERT INTO exams (title, description, duration) VALUES
('Sample Math Exam', 'A basic math proficiency test', 60);

-- Insert sample exam questions
INSERT INTO exam_questions (exam_name, question_number, question_text, question_type, options, correct_answer, points) VALUES
('Sample Math Exam', 1, 'What is 2 + 2?', 'multiple_choice', '["2", "3", "4", "5"]', '4', 1),
('Sample Math Exam', 2, 'What is the square root of 16?', 'multiple_choice', '["2", "3", "4", "5"]', '4', 1),
('Sample Math Exam', 3, 'Explain the Pythagorean theorem.', 'essay', '[]', 'The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.', 5);