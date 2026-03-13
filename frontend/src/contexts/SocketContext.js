import React, { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from './AuthContext';
import toast from 'react-hot-toast';

const SocketContext = createContext();

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
};

export const SocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const { user } = useAuth();

  useEffect(() => {
    if (user) {
      const token = localStorage.getItem('token');
      if (!token) return;

      // Initialize socket connection
      const newSocket = io(process.env.REACT_APP_WS_URL || 'ws://localhost:5000', {
        auth: {
          token
        },
        transports: ['websocket', 'polling']
      });

      // Connection event handlers
      newSocket.on('connect', () => {
        console.log('Socket connected');
        setConnected(true);
        toast.success('Connected to monitoring system');
      });

      newSocket.on('disconnect', (reason) => {
        console.log('Socket disconnected:', reason);
        setConnected(false);
        if (reason !== 'io client disconnect') {
          toast.error('Connection lost. Attempting to reconnect...');
        }
      });

      newSocket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        setConnected(false);
        toast.error('Failed to connect to monitoring system');
      });

      // Alert handlers
      newSocket.on('new-alert', (alert) => {
        console.log('New alert received:', alert);
        setAlerts(prev => {
          // Prevent duplicates by checking ID if available, or title+timestamp
          const isDuplicate = prev.some(a => (a.id && a.id === alert.id) || (a.title === alert.title && a.sessionId === alert.sessionId && Math.abs(new Date(a.createdAt || a.timestamp) - new Date(alert.createdAt || alert.timestamp)) < 2000));
          if (isDuplicate) return prev;
          return [alert, ...prev.slice(0, 49)];
        });
      });

      newSocket.on('session-alert', (alert) => {
        console.log('Session alert received:', alert);
        // Handle session-specific alerts
      });

      newSocket.on('alert-acknowledged', (data) => {
        console.log('Alert acknowledged:', data);
        setAlerts(prev =>
          prev.map(alert =>
            alert.id === data.alertId
              ? { ...alert, acknowledged: true, acknowledgedBy: data.acknowledgedBy }
              : alert
          )
        );
      });

      // User activity handlers
      newSocket.on('user-joined', (data) => {
        console.log('User joined session:', data);
      });

      newSocket.on('user-left', (data) => {
        console.log('User left session:', data);
      });

      newSocket.on('user-disconnected', (data) => {
        console.log('User disconnected:', data);
      });

      // Student monitoring handlers (for supervisors)
      newSocket.on('student-video-frame', (data) => {
        // Handle incoming video frames from students
        console.log('Received video frame from student:', data.studentId);
      });

      newSocket.on('student-audio-chunk', (data) => {
        // Handle incoming audio chunks from students
        console.log('Received audio chunk from student:', data.studentId);
      });

      newSocket.on('student-status', (data) => {
        console.log('Student status update:', data);
      });

      newSocket.on('student-connection-quality', (data) => {
        console.log('Student connection quality:', data);
      });

      // Supervisor communication handlers (for students)
      newSocket.on('supervisor-message', (data) => {
        console.log('Message from supervisor:', data);
        toast.info(`Supervisor: ${data.message}`, {
          duration: 8000,
        });
      });

      // Error handlers
      newSocket.on('error', (error) => {
        console.error('Socket error:', error);
        toast.error(error.message || 'Socket error occurred');
      });

      setSocket(newSocket);

      return () => {
        newSocket.close();
        setSocket(null);
        setConnected(false);
      };
    }
  }, [user]);

  // Socket utility functions
  const joinSession = (sessionId) => {
    if (socket) {
      socket.emit('join-session', sessionId);
    }
  };

  const leaveSession = (sessionId) => {
    if (socket) {
      socket.emit('leave-session', sessionId);
    }
  };

  const sendVideoFrame = (sessionId, frameData, timestamp) => {
    if (socket && connected) {
      socket.emit('video-frame', {
        sessionId,
        frameData,
        timestamp
      });
    }
  };

  const sendAudioChunk = (sessionId, audioData, timestamp) => {
    if (socket && connected) {
      socket.emit('audio-chunk', {
        sessionId,
        audioData,
        timestamp
      });
    }
  };

  const sendSystemStatus = (sessionId, status) => {
    if (socket && connected) {
      socket.emit('system-status', {
        sessionId,
        status
      });
    }
  };

  const sendSupervisorMessage = (sessionId, studentId, message) => {
    if (socket && connected) {
      socket.emit('supervisor-message', {
        sessionId,
        studentId,
        message
      });
    }
  };

  const acknowledgeAlert = (alertId, sessionId) => {
    if (socket && connected) {
      socket.emit('acknowledge-alert', {
        alertId,
        sessionId
      });
    }
  };

  const sendConnectionQuality = (sessionId, quality) => {
    if (socket && connected) {
      socket.emit('connection-quality', {
        sessionId,
        quality
      });
    }
  };

  const ping = () => {
    if (socket && connected) {
      socket.emit('ping');
    }
  };

  // Clear alerts
  const clearAlerts = () => {
    setAlerts([]);
  };

  // Remove specific alert
  const removeAlert = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const value = {
    socket,
    connected,
    alerts,
    joinSession,
    leaveSession,
    sendVideoFrame,
    sendAudioChunk,
    sendSystemStatus,
    sendSupervisorMessage,
    acknowledgeAlert,
    sendConnectionQuality,
    ping,
    clearAlerts,
    removeAlert
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
};