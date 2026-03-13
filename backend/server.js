const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const examRoutes = require('./routes/exam');
const alertRoutes = require('./routes/alerts');
const analysisRoutes = require('./routes/analysis');
const adminRoutes = require('./routes/admin');
const riskRoutes = require('./routes/risk');
const { authenticateToken } = require('./middleware/auth');
const { initializeDatabase } = require('./config/database');
const { initializeRedis } = require('./config/redis');
const socketHandler = require('./socket/socketHandler');
const logger = require('./utils/logger');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || "http://localhost:3000",
  credentials: true
}));

// Rate limiting
app.set('trust proxy', 1); // Trust first proxy
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  validate: { xForwardedForHeader: false } // Disable validation for local testing
});
app.use(limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Logging middleware
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.path} - ${req.ip}`);
  next();
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/exam', authenticateToken, examRoutes);
app.use('/api/alerts', authenticateToken, alertRoutes);
app.use('/api/analysis', authenticateToken, analysisRoutes);
app.use('/api/admin', authenticateToken, adminRoutes);
app.use('/api/risk', authenticateToken, riskRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Socket.IO handling
socketHandler(io);

// Initialize services
async function startServer() {
  try {
    await initializeDatabase();
    await initializeRedis();

    const PORT = process.env.PORT || 5000;
    server.listen(PORT, () => {
      logger.info(`Server running on port ${PORT}`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

module.exports = { app, server, io };