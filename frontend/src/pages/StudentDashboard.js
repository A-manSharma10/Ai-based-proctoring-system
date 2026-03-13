import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Play,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Camera as LucideCamera,
  Mic as LucideMic,
  Wifi as LucideWifi,
  User as LucideUser,
  FileText
} from 'lucide-react';
import toast from 'react-hot-toast';

const StudentDashboard = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await axios.get('/api/exam/sessions');
      setSessions(response.data.sessions);
    } catch (error) {
      console.error('Error fetching sessions:', error);
      toast.error('Failed to load exam sessions');
    } finally {
      setLoading(false);
    }
  };

  const startExam = async (sessionId) => {
    try {
      await axios.post(`/api/exam/session/${sessionId}/start`);
      toast.success('Exam session started');
      navigate(`/exam/${sessionId}`);
    } catch (error) {
      console.error('Error starting exam:', error);
      toast.error(error.response?.data?.error || 'Failed to start exam');
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'scheduled':
        return <Clock className="h-5 w-5 text-warning-500" />;
      case 'active':
        return <Play className="h-5 w-5 text-primary-500" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-success-500" />;
      case 'terminated':
        return <XCircle className="h-5 w-5 text-danger-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'scheduled':
        return 'bg-warning-100 text-warning-800';
      case 'active':
        return 'bg-primary-100 text-primary-800';
      case 'completed':
        return 'bg-success-100 text-success-800';
      case 'terminated':
        return 'bg-danger-100 text-danger-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const canStartExam = (session) => {
    return session.status === 'scheduled' || session.status === 'active';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-4xl font-black text-gray-900 tracking-tight">Student Dashboard</h1>
          <p className="text-gray-500 font-medium">Manage your exam sessions and system readiness</p>
        </div>
        <div className="flex items-center space-x-2 bg-white/50 backdrop-blur-sm p-2 rounded-2xl border border-white/40 shadow-sm">
          <div className="h-3 w-3 bg-success-500 rounded-full animate-pulse"></div>
          <span className="text-xs font-bold text-gray-600 uppercase tracking-wider">System Online</span>
        </div>
      </div>

      {/* Hero Welcome Section */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary-600 to-indigo-700 p-8 text-white shadow-2xl shadow-primary-500/20">
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="max-w-xl text-center md:text-left">
            <h2 className="text-3xl font-bold mb-3">Academic Excellence Starts Here</h2>
            <p className="text-primary-100 mb-6 leading-relaxed">
              Your examination portal is ready. Ensure your environment is secure and your hardware is verified before starting your scheduled sessions.
            </p>
            <div className="flex flex-wrap justify-center md:justify-start gap-4">
              <div className="bg-white/10 backdrop-blur-md rounded-xl px-4 py-2 border border-white/10">
                <p className="text-[10px] uppercase font-bold opacity-60">Next Scheduled</p>
                <p className="font-bold">{sessions.find(s => s.status === 'scheduled')?.exam_name || 'No upcoming exams'}</p>
              </div>
            </div>
          </div>
          <div className="hidden lg:block">
            <div className="h-32 w-32 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-xl border border-white/20">
              <Clock className="h-16 w-16 text-white opacity-80" />
            </div>
          </div>
        </div>
        {/* Abstract shapes for design */}
        <div className="absolute top-[-10%] right-[-5%] h-64 w-64 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-20%] left-[-5%] h-48 w-48 bg-primary-400/20 rounded-full blur-2xl"></div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Requirements and Info */}
        <div className="space-y-8">
          <div className="glass-card p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
              <AlertTriangle className="h-5 w-5 mr-2 text-warning-500" />
              Pre-Exam Protocol
            </h3>
            <div className="space-y-6">
              {[
                { title: 'Camera & Lighting', desc: 'Secure well-lit environment', icon: <LucideCamera className="h-4 w-4" /> },
                { title: 'Audio Isolation', desc: 'Minimal background noise', icon: <LucideMic className="h-4 w-4" /> },
                { title: 'Connectivity', desc: 'Plugged or high-charge battery', icon: <LucideWifi className="h-4 w-4" /> }
              ].map((req, i) => (
                <div key={i} className="flex items-start space-x-4 group">
                  <div className="h-10 w-10 rounded-xl bg-gray-50 flex items-center justify-center text-gray-500 group-hover:bg-primary-50 group-hover:text-primary-600 transition-colors">
                    {req.icon}
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-800">{req.title}</h4>
                    <p className="text-xs text-gray-500">{req.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white/50 backdrop-blur-md rounded-3xl p-6 border border-white/40 shadow-sm">
            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-4">Quick Stats</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-white shadow-sm border border-gray-100">
                <p className="text-2xl font-black text-primary-600">{sessions.filter(s => s.status === 'completed').length}</p>
                <p className="text-[10px] font-bold text-gray-500 uppercase">Passed</p>
              </div>
              <div className="p-4 rounded-2xl bg-white shadow-sm border border-gray-100">
                <p className="text-2xl font-black text-accent-600">{sessions.length}</p>
                <p className="text-[10px] font-bold text-gray-500 uppercase">Assigned</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Sessions List */}
        <div className="lg:col-span-2 space-y-6">
          <h3 className="text-xl font-bold text-gray-900 px-2 flex items-center justify-between">
            Available Sessions
            <span className="text-xs bg-primary-100 text-primary-700 px-3 py-1 rounded-full uppercase tracking-tighter">
              {sessions.filter(canStartExam).length} Active
            </span>
          </h3>

          {sessions.length === 0 ? (
            <div className="glass-card p-12 text-center">
              <div className="mx-auto h-20 w-20 bg-gray-50 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="h-10 w-10 text-gray-300" />
              </div>
              <p className="text-gray-500 font-medium">No active sessions assigned to your profile.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {sessions.map((session) => (
                <div key={session.id} className="group glass-card p-6 transition-all duration-300 hover:shadow-premium hover:-translate-y-1">
                  <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center space-x-3">
                        <div className={`p-3 rounded-2xl ${session.status === 'active' ? 'bg-primary-100 text-primary-600' :
                          session.status === 'scheduled' ? 'bg-warning-100 text-warning-600' :
                            'bg-gray-100 text-gray-500'
                          }`}>
                          {getStatusIcon(session.status)}
                        </div>
                        <div>
                          <h4 className="text-lg font-bold text-gray-900 group-hover:text-primary-600 transition-colors">
                            {session.exam_name}
                          </h4>
                          <div className="flex items-center space-x-2">
                            <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded-md ${getStatusColor(session.status)}`}>
                              {session.status}
                            </span>
                            <span className="text-xs text-gray-400 font-medium">Session ID: {session.id}</span>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                        <div className="flex items-center text-sm text-gray-500 font-medium">
                          <LucideUser className="h-4 w-4 mr-2 opacity-50" />
                          {session.supervisor_name || 'Auto-Proctored'}
                        </div>
                        <div className="flex items-center text-sm text-gray-500 font-medium">
                          <Clock className="h-4 w-4 mr-2 opacity-50" />
                          {new Date(session.start_time).toLocaleDateString()} at {new Date(session.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>

                    <div className="flex-shrink-0 w-full md:w-auto">
                      {canStartExam(session) ? (
                        <button
                          onClick={() => startExam(session.id)}
                          className="w-full md:w-auto btn btn-primary flex items-center justify-center space-x-3 px-8 py-3 rounded-2xl"
                        >
                          <Play className="h-5 w-5 fill-current" />
                          <span className="font-bold">{session.status === 'active' ? 'Resume' : 'Begin Exam'}</span>
                        </button>
                      ) : (
                        <div className="flex flex-col gap-2">
                          <div className="w-full md:w-auto text-center px-8 py-3 rounded-2xl bg-gray-50 text-gray-400 font-bold border border-gray-100 italic">
                            {session.status === 'completed' ? 'Archived' : 'Locked'}
                          </div>
                          {session.status === 'completed' && (
                            <button
                              onClick={() => navigate(`/report/${session.id}`)}
                              className="w-full md:w-auto btn bg-primary-100 text-primary-700 hover:bg-primary-200 flex items-center justify-center space-x-2 px-8 py-2 rounded-xl text-xs font-bold transition-all"
                            >
                              <FileText className="h-4 w-4" />
                              <span>View Report</span>
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;