const express = require('express');
const bcrypt = require('bcryptjs');
const Joi = require('joi');
const axios = require('axios');
const { query } = require('../config/database');
const { setSession, deleteSession } = require('../config/redis');
const { generateToken } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().min(6).required(),
  faceImage: Joi.string().optional()
});

const registerSchema = Joi.object({
  email: Joi.string().email().required(),
  name: Joi.string().min(2).max(100).required(),
  password: Joi.string().min(6).required(),
  role: Joi.string().valid('student', 'supervisor').default('student'),
  faceImage: Joi.string().optional()
});

// Register new user
router.post('/register', async (req, res) => {
  try {
    const { error, value } = registerSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email, name, password, role, faceImage } = value;

    // Check if user already exists
    const existingUser = await query('SELECT id FROM users WHERE email = ?', [email]);
    if (existingUser.length > 0) {
      return res.status(409).json({ error: 'User already exists' });
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 10);

    // Process face image if provided
    let faceEmbedding = null;
    if (faceImage) {
      try {
        const faceResponse = await axios.post('http://face_service:8000/register-face', {
          image: faceImage,
          user_id: email
        });
        faceEmbedding = Buffer.from(faceResponse.data.embedding);
      } catch (faceError) {
        logger.warn('Face registration failed:', faceError.message);
      }
    }

    // Create user
    const result = await query(
      'INSERT INTO users (email, name, password_hash, role, face_embedding) VALUES (?, ?, ?, ?, ?)',
      [email, name, passwordHash, role, faceEmbedding]
    );

    res.status(201).json({
      message: 'User registered successfully',
      userId: result.insertId,
      faceRegistered: !!faceEmbedding
    });
  } catch (error) {
    logger.error('Registration error:', error);
    res.status(500).json({ error: 'Registration failed' });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    const { error, value } = loginSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ error: error.details[0].message });
    }

    const { email, password, faceImage } = value;

    // Get user from database
    const users = await query('SELECT * FROM users WHERE email = ?', [email]);
    logger.info(`Login attempt for: ${email}. User found: ${users.length > 0}`);

    if (users.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const user = users[0];

    // Verify password
    const passwordValid = await bcrypt.compare(password, user.password_hash);
    logger.info(`Password valid for ${email}: ${passwordValid}`);

    if (!passwordValid) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify face if image provided and user has face embedding
    let faceVerified = true;
    if (faceImage && user.face_embedding) {
      if (process.env.USE_MOCK === 'true') {
        logger.info('Simulating face verification (SUCCESS) in MOCK mode');
        faceVerified = true;
      } else {
        try {
          const faceResponse = await axios.post('http://face_service:8000/verify-face', {
            image: faceImage,
            user_id: user.id
          });
          faceVerified = faceResponse.data.verified;
        } catch (faceError) {
          logger.warn('Face verification failed:', faceError.message);
          faceVerified = false;
        }
      }
    }

    if (!faceVerified) {
      return res.status(401).json({ error: 'Face verification failed' });
    }

    // Generate JWT token
    const tokenPayload = {
      userId: user.id,
      email: user.email,
      name: user.name,
      role: user.role
    };

    const token = generateToken(tokenPayload);

    // Store session in Redis
    await setSession(`session:${user.id}`, {
      userId: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      loginTime: new Date().toISOString()
    }, 24 * 60 * 60); // 24 hours

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role
      },
      faceVerified
    });
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({ error: 'Login failed' });
  }
});

// Logout user
router.post('/logout', async (req, res) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (token) {
      const jwt = require('jsonwebtoken');
      const decoded = jwt.decode(token);
      if (decoded && decoded.userId) {
        await deleteSession(`session:${decoded.userId}`);
      }
    }

    res.json({ message: 'Logged out successfully' });
  } catch (error) {
    logger.error('Logout error:', error);
    res.status(500).json({ error: 'Logout failed' });
  }
});

// Verify session
router.get('/verify', async (req, res) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-super-secret-jwt-key');

    const session = await getSession(`session:${decoded.userId}`);
    if (!session) {
      return res.status(401).json({ error: 'Session expired' });
    }

    res.json({ valid: true, user: decoded });
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

module.exports = router;