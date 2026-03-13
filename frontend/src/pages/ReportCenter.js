import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
    Clock, AlertCircle, User,
    Download, ChevronLeft, Activity,
    CheckCircle
} from 'lucide-react';
import toast from 'react-hot-toast';

const ShieldCheck = ({ className }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
        <path d="m9 12 2 2 4-4" />
    </svg>
);

const getSeverityColor = (severity) => {
    const m = { low: 'bg-blue-100 text-blue-700', medium: 'bg-yellow-100 text-yellow-700', high: 'bg-orange-100 text-orange-700', critical: 'bg-red-100 text-red-700' };
    return m[severity] || 'bg-gray-100 text-gray-700';
};

const ReportCenter = () => {
    const { sessionId } = useParams();
    const navigate = useNavigate();
    const [report, setReport] = useState(null);
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (sessionId) fetchReportData();
    }, [sessionId]);

    const fetchReportData = async () => {
        try {
            const [sessionRes, alertsRes] = await Promise.all([
                axios.get(`/api/exam/session/${sessionId}`),
                axios.get(`/api/alerts/session/${sessionId}`)
            ]);
            setReport(sessionRes.data.session);
            setAlerts(alertsRes.data.alerts || []);
        } catch (error) {
            console.error('Error fetching report:', error);
            toast.error('Failed to load session report');
        } finally {
            setLoading(false);
        }
    };

    const getDuration = () => {
        if (!report?.start_time) return 'N/A';
        const start = new Date(report.start_time);
        const end = report.end_time ? new Date(report.end_time) : new Date();
        const mins = Math.floor((end - start) / 60000);
        if (mins < 60) return `${mins} Mins`;
        return `${Math.floor(mins / 60)}h ${mins % 60}m`;
    };

    const getTrustScore = () => {
        if (alerts.length === 0) return { score: 98, label: 'Excellent', color: 'text-green-600' };
        const highRisk = alerts.filter(a => a.severity === 'critical' || a.severity === 'high').length;
        const medRisk = alerts.filter(a => a.severity === 'medium').length;
        const score = Math.max(0, 100 - highRisk * 15 - medRisk * 5 - alerts.length * 2);
        if (score >= 80) return { score, label: 'Good', color: 'text-green-600' };
        if (score >= 60) return { score, label: 'Moderate', color: 'text-yellow-600' };
        return { score, label: 'Low', color: 'text-red-600' };
    };

    const trust = getTrustScore();

    if (loading) return (
        <div className="flex h-screen items-center justify-center">
            <div className="text-center space-y-4">
                <div className="loading-spinner mx-auto" />
                <p className="text-gray-500 text-sm">Loading session report...</p>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen bg-[#F8FAFC] p-6 lg:p-10">
            <div className="max-w-7xl mx-auto space-y-10">
                {/* Header */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                    <div className="space-y-3">
                        <button onClick={() => navigate(-1)} className="flex items-center text-sm font-bold text-gray-500 hover:text-primary-600 transition-colors">
                            <ChevronLeft className="h-4 w-4 mr-1" /> Back to Dashboard
                        </button>
                        <div>
                            <h1 className="text-4xl font-black text-gray-900 tracking-tight">Session Audit Report</h1>
                            <p className="text-gray-400 font-mono text-sm tracking-widest uppercase mt-1">
                                Session #{sessionId} · {report?.exam_name}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={() => { window.print(); toast('Print dialog opened.', { icon: '🖨️' }); }}
                        className="btn btn-primary px-8 py-3 rounded-2xl flex items-center gap-3 shadow-xl shadow-primary-500/20">
                        <Download className="h-5 w-5" />
                        <span className="font-bold">Export Report</span>
                    </button>
                </div>

                {/* Summary Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                    {[
                        {
                            icon: <User className="h-5 w-5" />, color: 'bg-primary-50 text-primary-600',
                            label: 'Candidate', value: report?.student_name || 'Unknown'
                        },
                        {
                            icon: <Clock className="h-5 w-5" />, color: 'bg-green-50 text-green-600',
                            label: 'Duration', value: getDuration()
                        },
                        {
                            icon: <AlertCircle className="h-5 w-5" />, color: 'bg-red-50 text-red-600',
                            label: 'Integrity Alerts', value: String(alerts.length)
                        },
                        {
                            icon: <ShieldCheck className="h-5 w-5" />, color: 'bg-blue-50 text-blue-600',
                            label: 'Trust Score', value: `${trust.score}%`, valueClass: trust.color
                        },
                    ].map((card, i) => (
                        <div key={i} className="glass-card p-6 space-y-3">
                            <div className={`h-10 w-10 ${card.color} rounded-xl flex items-center justify-center`}>
                                {card.icon}
                            </div>
                            <p className="text-xs font-black text-gray-400 uppercase tracking-widest">{card.label}</p>
                            <p className={`text-xl font-black text-gray-900 ${card.valueClass || ''}`}>{card.value}</p>
                        </div>
                    ))}
                </div>

                {/* Session Info + Status Bar */}
                <div className="glass-card p-6">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 text-sm">
                        <div>
                            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Student Email</p>
                            <p className="font-semibold text-gray-800">{report?.student_email || '—'}</p>
                        </div>
                        <div>
                            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Supervisor</p>
                            <p className="font-semibold text-gray-800">{report?.supervisor_name || '—'}</p>
                        </div>
                        <div>
                            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Start Time</p>
                            <p className="font-semibold text-gray-800">{report?.start_time ? new Date(report.start_time).toLocaleString() : '—'}</p>
                        </div>
                        <div>
                            <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Status</p>
                            <span className={`px-3 py-1 rounded-full text-xs font-black uppercase ${report?.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                                }`}>{report?.status}</span>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
                    {/* Alert Timeline */}
                    <div className="lg:col-span-2 space-y-6">
                        <h3 className="text-2xl font-black text-gray-900 flex items-center gap-3">
                            <Activity className="h-6 w-6 text-primary-600" /> Incident Timeline
                            <span className="text-sm font-semibold text-gray-400 ml-auto">{alerts.length} event{alerts.length !== 1 ? 's' : ''}</span>
                        </h3>

                        {alerts.length === 0 ? (
                            <div className="glass-card p-12 text-center">
                                <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-3" />
                                <p className="font-bold text-gray-700">No violations detected</p>
                                <p className="text-sm text-gray-400 mt-1">This session completed with no integrity alerts.</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {alerts.map((alert, i) => (
                                    <div key={i} className={`glass-card p-6 flex items-start gap-5 border-l-4 ${alert.severity === 'critical' ? 'border-l-red-500' :
                                        alert.severity === 'high' ? 'border-l-orange-400' :
                                            'border-l-yellow-400'
                                        }`}>
                                        <div className={`h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0 ${alert.severity === 'critical' ? 'bg-red-50 text-red-600' :
                                            alert.severity === 'high' ? 'bg-orange-50 text-orange-600' :
                                                'bg-yellow-50 text-yellow-600'
                                            }`}>
                                            <AlertCircle className="h-5 w-5" />
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex justify-between items-start gap-3">
                                                <h4 className="font-bold text-gray-900">{alert.title}</h4>
                                                <span className="text-[10px] font-black text-gray-400 bg-gray-50 px-2 py-1 rounded-full flex-shrink-0">
                                                    {new Date(alert.created_at).toLocaleTimeString()}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-500 mt-1">{alert.description}</p>
                                            <div className="mt-3 flex items-center gap-3">
                                                <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded ${getSeverityColor(alert.severity)}`}>
                                                    {alert.severity}
                                                </span>
                                                {alert.confidence_score && (
                                                    <span className="text-[10px] font-bold text-gray-400">
                                                        Confidence: {(alert.confidence_score * 100).toFixed(0)}%
                                                    </span>
                                                )}
                                                {alert.resolved && (
                                                    <span className="flex items-center gap-1 text-[10px] font-bold text-green-600">
                                                        <CheckCircle className="h-3 w-3" /> Resolved
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                        {alert.metadata?.evidence && (
                                            <div className="flex-shrink-0 w-24 h-16 rounded-xl overflow-hidden border border-gray-100 shadow-sm cursor-zoom-in group relative" onClick={() => window.open(alert.metadata.evidence)}>
                                                <img src={alert.metadata.evidence} alt="Evidence" className="w-full h-full object-cover grayscale-[0.5] group-hover:grayscale-0 transition-all" />
                                                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <Activity className="h-4 w-4 text-white" />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Sidebar: integrity score + metadata */}
                    <div className="space-y-6">
                        {/* Trust Score Breakdown */}
                        <div className="glass-card p-6 space-y-5">
                            <h3 className="text-lg font-bold text-gray-900">Integrity Score</h3>
                            <div className="text-center">
                                <div className={`text-5xl font-black ${trust.color}`}>{trust.score}%</div>
                                <p className={`text-sm font-bold mt-1 ${trust.color}`}>{trust.label}</p>
                            </div>
                            <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full transition-all ${trust.score >= 80 ? 'bg-green-500' : trust.score >= 60 ? 'bg-yellow-400' : 'bg-red-500'
                                    }`} style={{ width: `${trust.score}%` }} />
                            </div>
                            <div className="space-y-2 text-xs">
                                {[
                                    { label: 'High-Risk Alerts', value: alerts.filter(a => a.severity === 'critical' || a.severity === 'high').length, bad: true },
                                    { label: 'Medium Alerts', value: alerts.filter(a => a.severity === 'medium').length, warn: true },
                                    { label: 'Low Alerts', value: alerts.filter(a => a.severity === 'low').length },
                                    { label: 'Resolved', value: alerts.filter(a => a.resolved).length, good: true },
                                ].map((item, i) => (
                                    <div key={i} className="flex justify-between items-center">
                                        <span className="text-gray-500 font-medium">{item.label}</span>
                                        <span className={`font-black ${item.bad && item.value > 0 ? 'text-red-600' : item.warn && item.value > 0 ? 'text-yellow-600' : item.good ? 'text-green-600' : 'text-gray-700'}`}>
                                            {item.value}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Proctor Certification */}
                        <div className="bg-gradient-to-br from-gray-900 to-primary-900 rounded-3xl p-7 text-white space-y-4 shadow-2xl">
                            <div className="flex items-center gap-2">
                                <ShieldCheck className="h-5 w-5 text-green-400" />
                                <h4 className="font-bold">Proctor Certification</h4>
                            </div>
                            <p className="text-xs text-gray-400 leading-relaxed">
                                This report is generated by the AI Proctoring System. All events are timestamped and logged for academic integrity review.
                            </p>
                            <div className="border-t border-white/10 pt-4 text-[10px] text-gray-500 font-mono">
                                Session ID: {sessionId}<br />
                                Generated: {new Date().toLocaleString()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ReportCenter;
