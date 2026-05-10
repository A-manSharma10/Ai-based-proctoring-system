import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Webcam from 'react-webcam';
import {
  Maximize, Minimize, AlertTriangle, Eye, Phone,
  Users, Volume2, Clock, LogOut, Activity
} from 'lucide-react';
import { useSocket } from '../contexts/SocketContext';
import ViolationPopup from './ViolationPopup';

const EnhancedExamSession = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const { socket, joinSession, leaveSession, sendVideoFrame } = useSocket();
  const webcamRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const timerRef = useRef(null);

  const [examStarted, setExamStarted] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(30 * 60); // 30 minutes
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [violations, setViolations] = useState([]);
  const [activePopup, setActivePopup] = useState(null);
  const [warningsLeft, setWarningsLeft] = useState(5);
  const [isTerminated, setIsTerminated] = useState(false);
  const [terminationReason, setTerminationReason] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [questions] = useState([
    { id: 1, text: 'What is 2 + 2?', options: ['2', '3', '4', '5'] },
    { id: 2, text: 'What is the capital of France?', options: ['London', 'Berlin', 'Paris', 'Madrid'] }
  ]);

  // Fullscreen handling
  const toggleFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      console.error('Fullscreen error:', error);
    }
  }, []);

  // Handle fullscreen change
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);

      if (!document.fullscreenElement && examStarted) {
        showViolation({
          type: 'browser',
          title: 'Fullscreen Exited',
          message: 'Please return to fullscreen mode immediately',
          severity: 'high',
          confidence: 1.0
        });
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, [examStarted]);

  // Socket listeners for real-time proctoring events
  useEffect(() => {
    if (!socket || !sessionId || !examStarted) return;

    joinSession(sessionId);

    socket.on('violation-warning', (alert) => {
      showViolation({
        type: alert.alertType,
        title: alert.title,
        message: alert.description,
        severity: alert.severity,
        confidence: alert.confidenceScore
      });
    });

    socket.on('session-terminated', (data) => {
      setIsTerminated(true);
      setTerminationReason(data.reason);
      setExamStarted(false);
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(err => console.error(err));
      }
    });

    return () => {
      socket.off('violation-warning');
      socket.off('session-terminated');
      leaveSession(sessionId);
    };
  }, [socket, sessionId, examStarted, joinSession, leaveSession]);

  // Show violation popup
  const showViolation = useCallback((violation) => {
    const newViolation = {
      ...violation,
      id: Date.now(),
      timestamp: new Date().toISOString()
    };

    setViolations(prev => [newViolation, ...prev].slice(0, 50));
    setActivePopup(newViolation);
    setWarningsLeft(prev => Math.max(0, prev - 1));

    // Log to backend
    axios.post(`/api/alerts`, {
      sessionId: parseInt(sessionId),
      alertType: violation.type,
      severity: violation.severity,
      title: violation.title,
      description: violation.message,
      confidenceScore: violation.confidence || 0.9,
      metadata: { timestamp: new Date().toISOString() }
    }).catch(err => console.error('Failed to log violation:', err));
  }, [sessionId]);

  // AI Detection - Face, Gaze, Object, Audio with WORKING MOCK DETECTION
  const analyzeFrame = useCallback(async () => {
    if (!webcamRef.current || !examStarted) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    // Send frame to supervisor via Socket.IO
    sendVideoFrame(sessionId, imageSrc, new Date().toISOString());

    // Send frame to backend for analysis
    try {
      await axios.post('/api/analysis/frame', {
        sessionId: parseInt(sessionId),
        frameData: imageSrc,
        services: ['face', 'object', 'behavior']
      });
    } catch (err) {
      console.error('Frame analysis analysis error:', err);
    }
  }, [examStarted, sessionId, sendVideoFrame]);

  // Start frame analysis
  useEffect(() => {
    if (!examStarted) return;

    // Check every 2 seconds for violations (more frequent for smooth monitoring)
    frameIntervalRef.current = setInterval(analyzeFrame, 2000);

    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, [examStarted, analyzeFrame]);

  // Timer countdown
  useEffect(() => {
    if (!examStarted) return;

    timerRef.current = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          handleSubmit();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [examStarted]);

  const startExam = async () => {
    await toggleFullscreen();
    setExamStarted(true);
  };

  const handleSubmit = () => {
    if (window.confirm('Submit exam?')) {
      navigate('/dashboard');
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isTerminated) {
    return (
      <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
        <div className="max-w-m w-full bg-white rounded-3xl shadow-2xl p-10 text-center border-t-8 border-red-600">
          <div className="h-20 w-20 bg-red-100 text-red-600 rounded-full flex items-center justify-center mx-auto mb-6">
            <AlertTriangle className="h-10 w-10" />
          </div>
          <h1 className="text-3xl font-black text-gray-900 mb-4">Exam Terminated</h1>
          <p className="text-gray-600 mb-8 font-medium">
            This session has been automatically closed due to multiple security violations.
            <br /><span className="text-red-600 font-bold mt-2 block">Reason: {terminationReason}</span>
          </p>
          <button
            onClick={() => navigate('/dashboard')}
            className="w-full bg-gray-900 text-white py-4 rounded-xl font-bold hover:bg-black transition-all"
          >
            Return to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!examStarted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="max-w-2xl w-full bg-white rounded-3xl shadow-2xl p-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Ready to Start Exam?</h1>

          <div className="space-y-4 mb-8">
            <div className="flex items-center gap-3 p-4 bg-blue-50 rounded-xl">
              <Eye className="h-6 w-6 text-blue-600" />
              <div>
                <p className="font-semibold text-gray-900">AI Proctoring Active</p>
                <p className="text-sm text-gray-600">Your face, gaze, and surroundings will be monitored</p>
              </div>
            </div>

            <div className="flex items-center gap-3 p-4 bg-yellow-50 rounded-xl">
              <AlertTriangle className="h-6 w-6 text-yellow-600" />
              <div>
                <p className="font-semibold text-gray-900">Warnings: {warningsLeft} remaining</p>
                <p className="text-sm text-gray-600">Violations will reduce your warning count</p>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="w-full rounded-xl"
              videoConstraints={{ width: 640, height: 480, facingMode: 'user' }}
            />
          </div>

          <button
            onClick={startExam}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-bold text-lg hover:from-blue-700 hover:to-indigo-700 transition-all shadow-lg"
          >
            Start Exam (Fullscreen)
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-16 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-gray-900">Exam Session</h1>
            <div className="flex items-center gap-2 px-3 py-1 bg-green-100 rounded-lg">
              <Activity className="h-4 w-4 text-green-600 animate-pulse" />
              <span className="text-sm font-semibold text-green-700">AI Monitoring Active</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Warnings */}
            <div className="flex items-center gap-2 px-3 py-1 bg-yellow-100 rounded-lg">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <span className="text-sm font-bold text-yellow-700">Warnings: {warningsLeft}</span>
            </div>

            {/* Timer */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg font-mono font-bold ${timeRemaining < 300 ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-700'
              }`}>
              <Clock className="h-4 w-4" />
              {formatTime(timeRemaining)}
            </div>

            {/* Fullscreen Toggle */}
            <button
              onClick={toggleFullscreen}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            >
              {isFullscreen ? <Minimize className="h-5 w-5" /> : <Maximize className="h-5 w-5" />}
            </button>

            {/* Submit */}
            <button
              onClick={handleSubmit}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-bold"
            >
              <LogOut className="h-4 w-4" />
              Submit
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-6 grid grid-cols-4 gap-6">
        {/* Sidebar - Webcam */}
        <aside className="col-span-1">
          <div className="bg-white rounded-xl shadow-lg p-4">
            <h3 className="text-sm font-bold text-gray-700 mb-3">Live Monitoring</h3>
            <div className="relative aspect-video rounded-lg overflow-hidden bg-black">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                className="w-full h-full object-cover"
                videoConstraints={{ width: 320, height: 240, facingMode: 'user' }}
              />
              <div className="absolute top-2 left-2 flex items-center gap-1 bg-red-500 px-2 py-1 rounded-full">
                <div className="h-2 w-2 bg-white rounded-full animate-pulse" />
                <span className="text-xs font-bold text-white">REC</span>
              </div>
            </div>

            {/* Recent Violations */}
            <div className="mt-4">
              <h4 className="text-xs font-bold text-gray-500 uppercase mb-2">Recent Alerts</h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {violations.slice(0, 5).map(v => (
                  <div key={v.id} className="text-xs p-2 bg-red-50 rounded-lg border border-red-200">
                    <p className="font-semibold text-red-700">{v.title}</p>
                    <p className="text-red-600 text-[10px]">{new Date(v.timestamp).toLocaleTimeString()}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Question Area */}
        <div className="col-span-3">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">
                  Question {currentQuestion + 1} of {questions.length}
                </h2>
              </div>

              <p className="text-lg text-gray-700 mb-6">
                {questions[currentQuestion].text}
              </p>

              <div className="space-y-3">
                {questions[currentQuestion].options.map((option, idx) => (
                  <button
                    key={idx}
                    className="w-full text-left p-4 border-2 border-gray-200 rounded-xl hover:border-blue-500 hover:bg-blue-50 transition-all"
                  >
                    <span className="font-semibold">{String.fromCharCode(65 + idx)}.</span> {option}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex justify-between mt-8">
              <button
                onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
                disabled={currentQuestion === 0}
                className="px-6 py-3 bg-gray-200 rounded-lg font-semibold disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentQuestion(prev => Math.min(questions.length - 1, prev + 1))}
                disabled={currentQuestion === questions.length - 1}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Violation Popup */}
      {activePopup && (
        <ViolationPopup
          violation={activePopup}
          warningsLeft={warningsLeft}
          onClose={() => setActivePopup(null)}
        />
      )}
    </div>
  );
};

export default EnhancedExamSession;
