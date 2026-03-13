import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useSocket } from '../contexts/SocketContext';
import { ArrowLeft, Video, VideoOff, AlertTriangle, User, ShieldAlert, Activity } from 'lucide-react';
import toast from 'react-hot-toast';

const LiveMonitoring = () => {
    const { sessionId } = useParams();
    const navigate = useNavigate();
    const { socket, joinSession, leaveSession, alerts: socketAlerts, acknowledgeAlert } = useSocket();
    const [studentFrame, setStudentFrame] = useState(null);
    const [studentInfo, setStudentInfo] = useState({ name: 'Waiting for connection...', lastUpdate: null });
    const [sessionAlerts, setSessionAlerts] = useState([]);
    const [riskScore, setRiskScore] = useState({ total_score: 0, severity: 'low' });
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    useEffect(() => {
        if (!socket || !sessionId) return;

        // Join the specific exam session room
        joinSession(sessionId);
        toast.success(`Joined live monitoring for session ${sessionId}`);

        const handleVideoFrame = (data) => {
            setStudentFrame(data.frameData);
            setStudentInfo({
                id: data.studentId,
                name: data.studentName,
                lastUpdate: new Date(data.timestamp)
            });
        };

        socket.on('student-video-frame', handleVideoFrame);

        return () => {
            leaveSession(sessionId);
            socket.off('student-video-frame', handleVideoFrame);
        };
    }, [socket, sessionId, joinSession, leaveSession]);

    useEffect(() => {
        // Filter alerts for this session
        const filtered = socketAlerts.filter(a => parseInt(a.sessionId || a.session_id) === parseInt(sessionId));
        setSessionAlerts(filtered);
        if (filtered.length > 0) {
            fetchRiskScore();
            // Pulse analysis indicator when alert arrives
            setIsAnalyzing(true);
            setTimeout(() => setIsAnalyzing(false), 2000);
        }
    }, [socketAlerts, sessionId]);

    const fetchRiskScore = async () => {
        try {
            const res = await axios.get(`/api/risk/session/${sessionId}/score`);
            setRiskScore(res.data);
        } catch (err) {
            console.error('Failed to fetch risk score');
        }
    };

    useEffect(() => {
        const interval = setInterval(fetchRiskScore, 10000); // Update risk every 10s
        return () => clearInterval(interval);
    }, [sessionId]);

    const timeSinceLastFrame = studentInfo.lastUpdate
        ? Math.round((new Date() - studentInfo.lastUpdate) / 1000)
        : 0;

    const isVideoActive = studentInfo.lastUpdate && timeSinceLastFrame < 10;

    return (
        <div className="space-y-6 animate-in fade-in duration-500 max-w-7xl mx-auto">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/dashboard')}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                    >
                        <ArrowLeft className="h-6 w-6 text-gray-600" />
                    </button>
                    <div>
                        <h1 className="text-3xl font-black text-gray-900">Live Feed</h1>
                        <p className="text-gray-500 font-medium">Session ID: {sessionId}</p>
                    </div>
                </div>

                <div className="flex items-center gap-6">
                    <div className="flex flex-col items-end">
                        <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Live Risk Index</span>
                        <div className="flex items-center gap-2">
                            <div className="h-1.5 w-24 bg-gray-100 rounded-full overflow-hidden">
                                <div className={`h-full transition-all duration-1000 ${riskScore.total_score > 70 ? 'bg-red-500' : riskScore.total_score > 40 ? 'bg-yellow-500' : 'bg-green-500'}`} style={{ width: `${riskScore.total_score}%` }} />
                            </div>
                            <span className={`text-sm font-black ${riskScore.total_score > 70 ? 'text-red-600' : riskScore.total_score > 40 ? 'text-yellow-600' : 'text-green-600'}`}>
                                {Math.round(riskScore.total_score)}%
                            </span>
                        </div>
                    </div>

                    <div className="flex items-center gap-3 bg-white px-4 py-2 rounded-xl shadow-sm border border-gray-100 h-10">
                        <div className="flex items-center gap-2">
                            <div className={`h-2.5 w-2.5 rounded-full ${isVideoActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                            <span className="text-sm font-bold text-gray-700">
                                {isVideoActive ? 'Receiving Stream' : 'Stream Offline'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Video Feed */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="glass-card overflow-hidden border-2 border-gray-100">
                        <div className="bg-gray-900 p-3 flex justify-between items-center border-b border-gray-800">
                            <div className="flex items-center gap-2 text-gray-200">
                                <User className="h-4 w-4" />
                                <span className="text-sm font-semibold">{studentInfo.name}</span>
                            </div>
                            <div className="flex items-center gap-2 text-xs font-bold font-mono">
                                {isVideoActive ? (
                                    <span className="text-green-400 flex items-center gap-1"><Video className="h-3.5 w-3.5" /> LIVE</span>
                                ) : (
                                    <span className="text-red-400 flex items-center gap-1"><VideoOff className="h-3.5 w-3.5" /> LATENCY: {timeSinceLastFrame}s</span>
                                )}
                            </div>
                        </div>

                        <div className="relative aspect-video bg-black flex items-center justify-center">
                            {studentFrame ? (
                                <img
                                    src={studentFrame}
                                    alt="Live student feed"
                                    className="w-full h-full object-contain"
                                />
                            ) : (
                                <div className="text-gray-600 flex flex-col items-center">
                                    <VideoOff className="h-16 w-16 mb-4 opacity-50" />
                                    <p className="font-medium text-lg">No video data received yet</p>
                                    <p className="text-sm opacity-70 mt-1">Waiting for student to enable camera...</p>
                                </div>
                            )}

                            {/* AI Analysis HUD Overlay */}
                            {isVideoActive && (
                                <div className="absolute inset-0 pointer-events-none p-6">
                                    <div className="h-full w-full border border-green-500/20 rounded-xl relative overflow-hidden">
                                        {/* Scanning Line */}
                                        <div className="absolute top-0 left-0 w-full h-[1px] bg-green-400/30 animate-scan pointer-events-none" />

                                        {/* HUD Corners */}
                                        <div className="absolute top-0 left-0 h-6 w-6 border-t-2 border-l-2 border-green-500/40 rounded-tl-lg" />
                                        <div className="absolute top-0 right-0 h-6 w-6 border-t-2 border-r-2 border-green-500/40 rounded-tr-lg" />
                                        <div className="absolute bottom-0 left-0 h-6 w-6 border-b-2 border-l-2 border-green-500/40 rounded-bl-lg" />
                                        <div className="absolute bottom-0 right-0 h-6 w-6 border-b-2 border-r-2 border-green-500/40 rounded-br-lg" />

                                        {/* AI Status */}
                                        <div className="absolute bottom-4 left-4 flex gap-3">
                                            {['FACE', 'IRIS', 'AUDIO', 'POSE'].map(s => (
                                                <div key={s} className="flex items-center gap-1.5">
                                                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                                                    <span className="text-[8px] font-black text-green-500/80 tracking-widest">{s}</span>
                                                </div>
                                            ))}
                                        </div>

                                        {isAnalyzing && (
                                            <div className="absolute top-4 left-4 bg-green-500/20 backdrop-blur-md px-3 py-1 rounded-full border border-green-500/30 flex items-center gap-2">
                                                <Activity className="h-3 w-3 text-green-400 animate-pulse" />
                                                <span className="text-[10px] font-black text-green-400 tracking-tighter italic">NEURAL ANALYSIS ACTIVE</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Overlay recent unacknowledged alerts as a quick flash */}
                            {sessionAlerts.filter(a => !a.resolved).slice(0, 1).map(alert => (
                                <div key={alert.id} className="absolute top-4 right-4 bg-red-600/90 text-white px-4 py-3 rounded-xl max-w-xs shadow-2xl backdrop-blur-md animate-bounce">
                                    <div className="flex items-center gap-2 font-black mb-1 text-sm">
                                        <AlertTriangle className="h-4 w-4" /> {alert.title}
                                    </div>
                                    <p className="text-xs opacity-90">{alert.description}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Sidebar Alerts */}
                <div className="glass-card p-5 h-[600px] flex flex-col">
                    <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-red-500" /> Live Flags
                        <span className="bg-red-100 text-red-700 py-0.5 px-2 rounded-full text-xs font-black">
                            {sessionAlerts.filter(safe => !safe.resolved).length} Pending
                        </span>
                    </h3>

                    <div className="flex-1 overflow-y-auto pr-2 space-y-3">
                        {sessionAlerts.length === 0 ? (
                            <div className="text-center py-10 text-gray-400">
                                <ShieldAlert className="h-10 w-10 mx-auto mb-2 opacity-30" />
                                <p className="font-medium text-sm">No integrity violations detected.</p>
                            </div>
                        ) : (
                            sessionAlerts.map(alert => (
                                <div key={alert.id} className={`p-4 rounded-xl border ${alert.resolved ? 'bg-gray-50 border-gray-100 opacity-60' : 'bg-red-50 border-red-100 shadow-sm'}`}>
                                    <div className="flex justify-between items-start mb-2">
                                        <span className={`text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full ${alert.severity === 'critical' ? 'bg-red-200 text-red-800' :
                                            alert.severity === 'high' ? 'bg-orange-200 text-orange-800' :
                                                'bg-yellow-200 text-yellow-800'
                                            }`}>
                                            {alert.severity}
                                        </span>
                                        <span className="text-[10px] text-gray-500 font-bold font-mono">
                                            {new Date(alert.created_at || Date.now()).toLocaleTimeString()}
                                        </span>
                                    </div>
                                    <p className="font-bold text-gray-900 text-sm mb-1">{alert.title}</p>
                                    <p className="text-xs text-gray-600 leading-relaxed mb-3">{alert.description}</p>

                                    {!alert.resolved && (
                                        <button
                                            onClick={() => acknowledgeAlert(alert.id, sessionId)}
                                            className="w-full py-2 bg-white border border-gray-200 hover:bg-gray-50 text-xs font-bold text-gray-700 rounded-lg transition-colors"
                                        >
                                            Acknowledge
                                        </button>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveMonitoring;
