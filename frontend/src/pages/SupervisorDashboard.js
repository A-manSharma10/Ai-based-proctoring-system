import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useSocket } from '../contexts/SocketContext';
import {
  Users, AlertTriangle, Eye, Clock, CheckCircle, XCircle,
  Bell, Filter, Search, Activity, ShieldAlert, Video,
  AlertCircle, User, FileText, Play, ChevronRight
} from 'lucide-react';
import toast from 'react-hot-toast';

const SupervisorDashboard = () => {
  const [sessions, setSessions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [alertsLoading, setAlertsLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filter, setFilter] = useState('all');
  const [sessionFilter, setSessionFilter] = useState('all');
  const navigate = useNavigate();
  const { alerts: socketAlerts, acknowledgeAlert, connected } = useSocket();

  useEffect(() => {
    fetchSessions();
    fetchAlerts();
  }, []);

  useEffect(() => {
    if (socketAlerts.length > 0) {
      setAlerts(prev => {
        const newAlerts = socketAlerts.filter(sa => !prev.some(a => a.id === sa.id));
        if (newAlerts.length > 0) toast.error(`🚨 ${newAlerts.length} new alert(s)!`, { duration: 4000 });
        return [...newAlerts, ...prev];
      });
    }
  }, [socketAlerts]);

  const fetchSessions = async () => {
    try {
      const res = await axios.get('/api/exam/sessions');
      setSessions(res.data.sessions || []);
    } catch (err) {
      toast.error('Failed to load sessions');
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await axios.get('/api/alerts/supervisor');
      setAlerts(res.data.alerts || []);
    } catch (err) {
      console.error('Alerts fetch error:', err);
    } finally {
      setAlertsLoading(false);
    }
  };

  const handleAcknowledgeAlert = async (alertId, sessionId) => {
    try {
      await axios.patch(`/api/alerts/${alertId}/resolve`, { resolved: true });
      acknowledgeAlert(alertId, sessionId);
      setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, resolved: true } : a));
      toast.success('Alert acknowledged');
    } catch {
      toast.error('Failed to acknowledge alert');
    }
  };

  const getStatusBadge = (status) => {
    const map = {
      scheduled: { label: 'Scheduled', cls: 'bg-yellow-100 text-yellow-800' },
      active: { label: 'Live', cls: 'bg-green-100 text-green-700' },
      completed: { label: 'Completed', cls: 'bg-blue-100 text-blue-800' },
      terminated: { label: 'Terminated', cls: 'bg-red-100 text-red-800' },
    };
    const s = map[status] || { label: status, cls: 'bg-gray-100 text-gray-700' };
    return <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider ${s.cls}`}>{s.label}</span>;
  };

  const getSeverityColor = (severity) => {
    const m = { low: 'bg-blue-100 text-blue-700', medium: 'bg-yellow-100 text-yellow-700', high: 'bg-orange-100 text-orange-700', critical: 'bg-red-100 text-red-700' };
    return m[severity] || 'bg-gray-100 text-gray-700';
  };

  const filteredAlerts = alerts.filter(a => {
    const matchFilter = filter === 'all' || a.severity === filter;
    const matchSearch = !searchTerm || a.student_name?.toLowerCase().includes(searchTerm.toLowerCase()) || a.title.toLowerCase().includes(searchTerm.toLowerCase());
    return matchFilter && matchSearch;
  });

  const filteredSessions = sessions.filter(s =>
    sessionFilter === 'all' || s.status === sessionFilter
  );

  const stats = {
    activeSessions: sessions.filter(s => s.status === 'active').length,
    completedSessions: sessions.filter(s => s.status === 'completed').length,
    totalAlerts: alerts.length,
    unresolvedAlerts: alerts.filter(a => !a.resolved).length,
  };

  if (loading) return <div className="flex items-center justify-center min-h-[400px]"><div className="loading-spinner" /></div>;

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-4xl font-black text-gray-900 tracking-tight">Supervision Console</h1>
          <p className="text-gray-500 font-medium">Real-time integrity monitoring and session management</p>
        </div>
        <div className="flex items-center gap-2 bg-white/50 backdrop-blur-sm px-4 py-2 rounded-2xl border border-white/40 shadow-sm">
          <div className={`h-2 w-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">{connected ? 'Live Sync' : 'Offline'}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
        {[
          { label: 'Live Sessions', value: stats.activeSessions, icon: <Video className="h-6 w-6" />, color: 'bg-green-50 text-green-600' },
          { label: 'Completed', value: stats.completedSessions, icon: <CheckCircle className="h-6 w-6" />, color: 'bg-blue-50 text-blue-600' },
          { label: 'Total Alerts', value: stats.totalAlerts, icon: <AlertTriangle className="h-6 w-6" />, color: 'bg-red-50 text-red-600' },
          { label: 'Unresolved', value: stats.unresolvedAlerts, icon: <ShieldAlert className="h-6 w-6" />, color: 'bg-yellow-50 text-yellow-600' },
        ].map((s, i) => (
          <div key={i} className="glass-card p-6 flex items-center gap-4">
            <div className={`p-3 rounded-2xl ${s.color}`}>{s.icon}</div>
            <div>
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">{s.label}</p>
              <p className="text-3xl font-black text-gray-900">{s.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Sessions Table */}
      <div className="glass-card overflow-hidden">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 p-6 border-b border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Users className="h-5 w-5 text-primary-500" /> All Sessions
          </h3>
          <div className="flex gap-2">
            {['all', 'active', 'scheduled', 'completed'].map(f => (
              <button key={f} onClick={() => setSessionFilter(f)}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold capitalize transition-colors ${sessionFilter === f ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                  }`}>{f}</button>
            ))}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-gray-50/70 text-[10px] font-black text-gray-400 uppercase tracking-widest">
              <tr>
                <th className="px-6 py-4">Student</th>
                <th className="px-6 py-4">Exam</th>
                <th className="px-6 py-4">Start Time</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {filteredSessions.length === 0 ? (
                <tr><td colSpan="5" className="px-6 py-12 text-center text-gray-400 italic">No sessions found.</td></tr>
              ) : filteredSessions.map(session => (
                <tr key={session.id} className="hover:bg-primary-50/20 transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="h-8 w-8 bg-primary-100 rounded-xl flex items-center justify-center">
                        <User className="h-4 w-4 text-primary-600" />
                      </div>
                      <div>
                        <p className="text-sm font-bold text-gray-900">{session.student_name}</p>
                        <p className="text-xs text-gray-400">{session.student_email}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-sm font-bold text-gray-800">{session.exam_name}</p>
                    <p className="text-xs text-gray-400">ID: {session.id}</p>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-sm text-gray-600">{new Date(session.start_time).toLocaleDateString()}</p>
                    <p className="text-xs text-gray-400">{new Date(session.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                  </td>
                  <td className="px-6 py-4">{getStatusBadge(session.status)}</td>
                  <td className="px-6 py-4 text-right">
                    {session.status === 'completed' ? (
                      <button
                        onClick={() => navigate(`/report/${session.id}`)}
                        className="flex items-center gap-1.5 ml-auto px-4 py-2 bg-primary-600 text-white text-xs font-bold rounded-xl hover:bg-primary-700 transition-colors shadow-sm"
                      >
                        <FileText className="h-3.5 w-3.5" /> View Report
                      </button>
                    ) : session.status === 'active' ? (
                      <button
                        onClick={() => navigate(`/live/${session.id}`)}
                        className="flex items-center justify-end gap-1.5 ml-auto px-4 py-2 bg-green-50 hover:bg-green-100 text-green-700 text-xs font-bold rounded-xl transition-colors shadow-sm"
                      >
                        <div className="h-1.5 w-1.5 bg-green-500 rounded-full animate-pulse" /> Watch Live
                      </button>
                    ) : (
                      <span className="text-xs text-gray-400 font-medium">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Alerts */}
      <div className="space-y-5">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-red-500" /> Security Alerts
            {stats.unresolvedAlerts > 0 && (
              <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs font-black rounded-full">{stats.unresolvedAlerts} pending</span>
            )}
          </h3>
          <div className="flex gap-2 w-full sm:w-auto">
            <div className="relative flex-1 sm:flex-none">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input type="text" placeholder="Search alerts..." className="input pl-10 h-10 text-sm py-0 w-full rounded-2xl"
                value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
            </div>
            <select className="input h-10 py-0 text-sm min-w-[120px] rounded-2xl" value={filter} onChange={e => setFilter(e.target.value)}>
              <option value="all">All Levels</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-gray-50/70 text-[10px] font-black text-gray-400 uppercase tracking-widest border-b border-gray-100">
                <tr>
                  <th className="px-6 py-4">Severity</th>
                  <th className="px-6 py-4">Event</th>
                  <th className="px-6 py-4">Student</th>
                  <th className="px-6 py-4">Time</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {alertsLoading ? (
                  <tr><td colSpan="6" className="px-6 py-12 text-center"><div className="loading-spinner mx-auto" /></td></tr>
                ) : filteredAlerts.length === 0 ? (
                  <tr><td colSpan="6" className="px-6 py-12 text-center text-gray-400 italic font-medium">
                    {alerts.length === 0 ? 'No alerts recorded yet. Alerts will appear here during active exams.' : 'No alerts match your filters.'}
                  </td></tr>
                ) : filteredAlerts.map(alert => (
                  <tr key={alert.id} className="hover:bg-red-50/20 transition-colors">
                    <td className="px-6 py-4">
                      <span className={`px-2.5 py-1 rounded-lg text-[10px] font-black uppercase ${getSeverityColor(alert.severity)}`}>
                        {alert.severity}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-sm font-bold text-gray-900">{alert.title}</p>
                      <p className="text-xs text-gray-500 max-w-xs truncate">{alert.description}</p>
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-sm font-bold text-gray-700">{alert.student_name}</p>
                      <p className="text-[10px] text-gray-400 uppercase font-bold">{alert.exam_name}</p>
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-xs text-gray-500">{new Date(alert.created_at).toLocaleTimeString()}</p>
                    </td>
                    <td className="px-6 py-4">
                      {alert.resolved ? (
                        <span className="flex items-center gap-1 text-xs text-green-600 font-bold"><CheckCircle className="h-3.5 w-3.5" /> Handled</span>
                      ) : (
                        <span className="flex items-center gap-1 text-xs text-yellow-600 font-bold"><AlertTriangle className="h-3.5 w-3.5" /> Pending</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {!alert.resolved && (
                          <button onClick={() => handleAcknowledgeAlert(alert.id, alert.session_id)}
                            className="px-3 py-1.5 bg-primary-600 text-white text-[10px] font-black uppercase rounded-lg hover:bg-primary-700 transition-colors">
                            Acknowledge
                          </button>
                        )}
                        <button onClick={() => navigate(`/report/${alert.session_id}`)}
                          className="px-3 py-1.5 bg-gray-100 text-gray-600 text-[10px] font-black uppercase rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-1">
                          <FileText className="h-3 w-3" /> Report
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SupervisorDashboard;