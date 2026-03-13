const jwt = require('jsonwebtoken');
const { getSession } = require('../config/redis');
const logger = require('../utils/logger');

const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key';

// Store active connections
const activeConnections = new Map();
const sessionConnections = new Map(); // sessionId -> Set of socket IDs

function socketHandler(io) {
  // Authentication middleware for Socket.IO
  io.use(async (socket, next) => {
    try {
      const token = socket.handshake.auth.token;
      if (!token) {
        return next(new Error('Authentication error: No token provided'));
      }

      const decoded = jwt.verify(token, JWT_SECRET);
      const session = await getSession(`session:${decoded.userId}`);

      if (!session) {
        return next(new Error('Authentication error: Session expired'));
      }

      socket.userId = decoded.userId;
      socket.userRole = decoded.role;
      socket.userName = decoded.name;

      next();
    } catch (error) {
      logger.error('Socket authentication error:', error);
      next(new Error('Authentication error: Invalid token'));
    }
  });

  io.on('connection', (socket) => {
    logger.info(`User ${socket.userName} (${socket.userId}) connected via Socket.IO`);

    // Store active connection
    activeConnections.set(socket.id, {
      userId: socket.userId,
      userRole: socket.userRole,
      userName: socket.userName,
      connectedAt: new Date()
    });

    // Join exam session room
    socket.on('join-session', (sessionId) => {
      try {
        socket.join(`session_${sessionId}`);
        socket.currentSession = sessionId;

        // Track session connections
        if (!sessionConnections.has(sessionId)) {
          sessionConnections.set(sessionId, new Set());
        }
        sessionConnections.get(sessionId).add(socket.id);

        logger.info(`User ${socket.userName} joined session ${sessionId}`);

        // Notify others in the session
        socket.to(`session_${sessionId}`).emit('user-joined', {
          userId: socket.userId,
          userName: socket.userName,
          userRole: socket.userRole
        });

        // Send current session participants to the new user
        const sessionParticipants = Array.from(sessionConnections.get(sessionId))
          .map(socketId => activeConnections.get(socketId))
          .filter(conn => conn && conn.userId !== socket.userId);

        socket.emit('session-participants', sessionParticipants);
      } catch (error) {
        logger.error('Join session error:', error);
        socket.emit('error', { message: 'Failed to join session' });
      }
    });

    // Leave exam session room
    socket.on('leave-session', (sessionId) => {
      try {
        socket.leave(`session-${sessionId}`);

        // Remove from session connections
        if (sessionConnections.has(sessionId)) {
          sessionConnections.get(sessionId).delete(socket.id);
          if (sessionConnections.get(sessionId).size === 0) {
            sessionConnections.delete(sessionId);
          }
        }

        socket.currentSession = null;
        logger.info(`User ${socket.userName} left session ${sessionId}`);

        // Notify others in the session
        socket.to(`session_${sessionId}`).emit('user-left', {
          userId: socket.userId,
          userName: socket.userName
        });
      } catch (error) {
        logger.error('Leave session error:', error);
      }
    });

    // Handle video stream data
    socket.on('video-frame', (data) => {
      try {
        const { sessionId, frameData, timestamp } = data;

        if (socket.userRole !== 'student') {
          return socket.emit('error', { message: 'Only students can send video frames' });
        }

        // Forward frame to supervisors in the same session
        socket.to(`session_${sessionId}`).emit('student-video-frame', {
          studentId: socket.userId,
          studentName: socket.userName,
          frameData,
          timestamp
        });

        // Here you could also trigger AI analysis
        // This would typically be done via HTTP API call to analysis service

      } catch (error) {
        logger.error('Video frame handling error:', error);
      }
    });

    // Handle audio stream data
    socket.on('audio-chunk', (data) => {
      try {
        const { sessionId, audioData, timestamp } = data;

        if (socket.userRole !== 'student') {
          return socket.emit('error', { message: 'Only students can send audio data' });
        }

        // Forward audio to supervisors in the same session
        socket.to(`session_${sessionId}`).emit('student-audio-chunk', {
          studentId: socket.userId,
          studentName: socket.userName,
          audioData,
          timestamp
        });

      } catch (error) {
        logger.error('Audio chunk handling error:', error);
      }
    });

    // Handle system status updates
    socket.on('system-status', (data) => {
      try {
        const { sessionId, status } = data;

        // Forward status to supervisors
        socket.to(`session_${sessionId}`).emit('student-status', {
          studentId: socket.userId,
          studentName: socket.userName,
          status,
          timestamp: new Date()
        });

      } catch (error) {
        logger.error('System status handling error:', error);
      }
    });

    // Handle supervisor messages to students
    socket.on('supervisor-message', (data) => {
      try {
        const { sessionId, studentId, message } = data;

        if (socket.userRole !== 'supervisor' && socket.userRole !== 'admin') {
          return socket.emit('error', { message: 'Only supervisors can send messages' });
        }

        // Send message to specific student
        const targetSockets = Array.from(sessionConnections.get(sessionId) || [])
          .map(socketId => activeConnections.get(socketId))
          .filter(conn => conn && conn.userId === studentId);

        targetSockets.forEach(conn => {
          io.to(conn.socketId).emit('supervisor-message', {
            supervisorName: socket.userName,
            message,
            timestamp: new Date()
          });
        });

      } catch (error) {
        logger.error('Supervisor message handling error:', error);
      }
    });

    // Handle alert acknowledgment
    socket.on('acknowledge-alert', (data) => {
      try {
        const { alertId, sessionId } = data;

        if (socket.userRole !== 'supervisor' && socket.userRole !== 'admin') {
          return socket.emit('error', { message: 'Only supervisors can acknowledge alerts' });
        }

        // Broadcast alert acknowledgment to session
        socket.to(`session-${sessionId}`).emit('alert-acknowledged', {
          alertId,
          acknowledgedBy: socket.userName,
          timestamp: new Date()
        });

      } catch (error) {
        logger.error('Alert acknowledgment handling error:', error);
      }
    });

    // Handle connection quality updates
    socket.on('connection-quality', (data) => {
      try {
        const { sessionId, quality } = data;

        // Forward connection quality to supervisors
        socket.to(`session-${sessionId}`).emit('student-connection-quality', {
          studentId: socket.userId,
          studentName: socket.userName,
          quality,
          timestamp: new Date()
        });

      } catch (error) {
        logger.error('Connection quality handling error:', error);
      }
    });

    // Handle ping/pong for connection monitoring
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: new Date() });
    });

    // Handle disconnection
    socket.on('disconnect', (reason) => {
      logger.info(`User ${socket.userName} (${socket.userId}) disconnected: ${reason}`);

      // Remove from active connections
      activeConnections.delete(socket.id);

      // Remove from session connections
      if (socket.currentSession && sessionConnections.has(socket.currentSession)) {
        sessionConnections.get(socket.currentSession).delete(socket.id);
        if (sessionConnections.get(socket.currentSession).size === 0) {
          sessionConnections.delete(socket.currentSession);
        }

        // Notify others in the session
        socket.to(`session-${socket.currentSession}`).emit('user-disconnected', {
          userId: socket.userId,
          userName: socket.userName,
          reason
        });
      }
    });

    // Handle errors
    socket.on('error', (error) => {
      logger.error(`Socket error for user ${socket.userName}:`, error);
    });
  });

  // Broadcast alert to all supervisors
  io.broadcastAlert = (alert) => {
    try {
      // Send to all supervisors and admins
      const supervisorConnections = Array.from(activeConnections.values())
        .filter(conn => conn.userRole === 'supervisor' || conn.userRole === 'admin');

      supervisorConnections.forEach(conn => {
        io.to(conn.socketId).emit('new-alert', alert);
      });

      // Also send to session-specific room
      if (alert.sessionId) {
        io.to(`session_${alert.sessionId}`).emit('session-alert', alert);
      }
    } catch (error) {
      logger.error('Broadcast alert error:', error);
    }
  };

  // Get active connections for monitoring
  io.getActiveConnections = () => {
    return {
      total: activeConnections.size,
      connections: Array.from(activeConnections.values()),
      sessions: Object.fromEntries(
        Array.from(sessionConnections.entries()).map(([sessionId, socketIds]) => [
          sessionId,
          Array.from(socketIds).map(socketId => activeConnections.get(socketId)).filter(Boolean)
        ])
      )
    };
  };

  logger.info('Socket.IO handler initialized');
}

module.exports = socketHandler;