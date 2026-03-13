import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Users,
    BookOpen,
    Activity,
    Plus,
    Trash2,
    Edit3,
    ShieldCheck,
    Search,
    LayoutDashboard,
    Settings,
    ChevronRight,
    Monitor,
    Cpu,
    BarChart3,
    Trophy,
    ShieldAlert
} from 'lucide-react';
import toast from 'react-hot-toast';

const AdminDashboard = () => {
    const [stats, setStats] = useState({ users: 0, exams: 0, activeSessions: 0 });
    const [users, setUsers] = useState([]);
    const [exams, setExams] = useState([]);
    const [activeTab, setActiveTab] = useState('overview');
    const [loading, setLoading] = useState(true);
    const [showDeployModal, setShowDeployModal] = useState(false);
    const [health, setHealth] = useState({});
    const [metrics, setMetrics] = useState(null);

    // Modal state
    const [examForm, setExamForm] = useState({
        title: '', description: '', duration: 60, start_time: '', end_time: '',
        questions: [], supervisors: [], students: []
    });

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const [statsRes, usersRes, examsRes, healthRes, metricsRes] = await Promise.all([
                axios.get('/api/admin/stats'),
                axios.get('/api/admin/users'),
                axios.get('/api/exam/exams'),
                axios.get('/api/admin/health'),
                axios.get('/api/admin/metrics')
            ]);
            setStats(statsRes.data);
            setUsers(usersRes.data.users);
            setExams(examsRes.data);
            setHealth(healthRes.data);
            setMetrics(metricsRes.data);
        } catch (error) {
            console.error('Error fetching admin data:', error);
            toast.error('Failed to load administrative data');
        } finally {
            setLoading(false);
        }
    };

    const SidebarItem = ({ id, icon: Icon, label }) => (
        <button
            onClick={() => setActiveTab(id)}
            className={`w-full flex items-center space-x-3 px-6 py-4 transition-all ${activeTab === id
                ? 'bg-primary-50 text-primary-600 border-r-4 border-primary-600 font-bold'
                : 'text-gray-500 hover:bg-gray-50'
                }`}
        >
            <Icon className="h-5 w-5" />
            <span>{label}</span>
        </button>
    );

    const handleDeploySubmit = async (e) => {
        e.preventDefault();
        try {
            // 1. Create Exam
            const { data: { id: examId } } = await axios.post('/api/admin/exams', {
                title: examForm.title,
                description: examForm.description,
                duration: examForm.duration,
                start_time: examForm.start_time,
                end_time: examForm.end_time
            });
            // 2. Add Questions
            for (let i = 0; i < examForm.questions.length; i++) {
                const q = examForm.questions[i];
                await axios.post('/api/admin/questions', {
                    examName: examForm.title,
                    questionNumber: i + 1,
                    questionText: q.text,
                    questionType: q.type,
                    options: q.options || [],
                    points: q.points || 1
                });
            }
            // 3. Assign Students (if valid IDs entered)
            if (examForm.students.length > 0 && examForm.supervisors.length > 0) {
                await axios.post('/api/admin/assign', {
                    examId, examName: examForm.title,
                    studentIds: examForm.students,
                    supervisorId: examForm.supervisors[0],
                    startTime: examForm.start_time
                });
            }
            toast.success('Exam successfully deployed!');
            setShowDeployModal(false);
            fetchData();
        } catch (error) {
            toast.error('Failed to deploy exam');
        }
    };

    if (loading) return <div className="flex h-screen items-center justify-center"><div className="loading-spinner"></div></div>;

    return (
        <div className="flex h-screen bg-[#F8FAFC]">
            {/* Sidebar */}
            <aside className="w-64 bg-white border-r border-gray-100 flex flex-col">
                <div className="p-8">
                    <div className="flex items-center space-x-3">
                        <div className="h-10 w-10 bg-primary-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-primary-200">
                            <ShieldCheck className="h-6 w-6" />
                        </div>
                        <span className="text-xl font-black text-gray-900 tracking-tight">Admin OS</span>
                    </div>
                </div>

                <nav className="flex-1 mt-4">
                    <SidebarItem id="overview" icon={LayoutDashboard} label="Overview" />
                    <SidebarItem id="users" icon={Users} label="User Registry" />
                    <SidebarItem id="exams" icon={BookOpen} label="Exam Manager" />
                    <SidebarItem id="settings" icon={Settings} label="System Config" />
                </nav>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto p-12">
                <header className="mb-12 flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-black text-gray-900">Control Center</h1>
                        <p className="text-gray-500 font-medium">Global system status and infrastructure management</p>
                    </div>
                    <div className="flex space-x-4">
                        <button className="btn btn-secondary flex items-center space-x-2">
                            <Search className="h-4 w-4" />
                            <span>Search Registry</span>
                        </button>
                        <button onClick={() => setShowDeployModal(true)} className="btn btn-primary flex items-center space-x-2 px-6">
                            <Plus className="h-4 w-4" />
                            <span>Deploy New Exam</span>
                        </button>
                    </div>
                </header>

                {/* Deploy Modal */}
                {showDeployModal && (
                    <div className="fixed inset-0 z-50 bg-gray-900/50 flex items-center justify-center p-4">
                        <div className="glass-card w-full max-w-3xl max-h-[90vh] overflow-y-auto p-8">
                            <h2 className="text-2xl font-black text-gray-900 mb-6">Deploy New Exam</h2>
                            <form onSubmit={handleDeploySubmit} className="space-y-6 text-sm">
                                <div className="grid grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">Exam Title</label>
                                        <input required type="text" className="w-full border-gray-200 rounded-xl p-3" value={examForm.title} onChange={e => setExamForm({ ...examForm, title: e.target.value })} placeholder="e.g. Midterm Computer Science" />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">Duration (mins)</label>
                                        <input required type="number" className="w-full border-gray-200 rounded-xl p-3" value={examForm.duration} onChange={e => setExamForm({ ...examForm, duration: e.target.value })} />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="font-bold text-gray-700">Description</label>
                                    <textarea className="w-full border-gray-200 rounded-xl p-3" value={examForm.description} onChange={e => setExamForm({ ...examForm, description: e.target.value })} />
                                </div>
                                <div className="grid grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">Start Time</label>
                                        <input required type="datetime-local" className="w-full border-gray-200 rounded-xl p-3" value={examForm.start_time} onChange={e => setExamForm({ ...examForm, start_time: e.target.value })} />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">End Time</label>
                                        <input required type="datetime-local" className="w-full border-gray-200 rounded-xl p-3" value={examForm.end_time} onChange={e => setExamForm({ ...examForm, end_time: e.target.value })} />
                                    </div>
                                </div>

                                <div className="space-y-4 border-t pt-6">
                                    <h3 className="font-black text-gray-800">Assignments</h3>
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">Assign Student IDs (comma separated)</label>
                                        <input type="text" placeholder="e.g. 1, 2, 5" className="w-full border-gray-200 rounded-xl p-3" onChange={e => setExamForm({ ...examForm, students: e.target.value.split(',').map(s => parseInt(s.trim())).filter(s => !isNaN(s)) })} />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="font-bold text-gray-700">Assign Supervisor ID</label>
                                        <input type="number" placeholder="Supervisor ID, e.g. 3" className="w-full border-gray-200 rounded-xl p-3" onChange={e => setExamForm({ ...examForm, supervisors: [parseInt(e.target.value)] })} />
                                    </div>
                                </div>

                                <div className="space-y-4 border-t pt-6">
                                    <div className="flex justify-between items-center">
                                        <h3 className="font-black text-gray-800">Questions ({examForm.questions.length})</h3>
                                        <button type="button" onClick={() => setExamForm({ ...examForm, questions: [...examForm.questions, { text: '', type: 'multiple_choice', options: ['A', 'B', 'C', 'D'], points: 1 }] })} className="text-xs font-bold text-primary-600 bg-primary-50 px-3 py-1.5 rounded-lg">+ Add Question</button>
                                    </div>
                                    {/* Map questions here... hidden for brevity but state holds them */}
                                    <button type="button" onClick={() => toast('CSV Import not configured yet', { icon: 'ℹ️' })} className="text-xs font-bold text-gray-600 bg-gray-100 px-3 py-1.5 rounded-lg">Import from CSV</button>
                                </div>

                                <div className="flex justify-end space-x-4 pt-6">
                                    <button type="button" onClick={() => setShowDeployModal(false)} className="px-6 py-2.5 font-bold text-gray-500 hover:bg-gray-50 rounded-xl transition-colors">Cancel</button>
                                    <button type="submit" className="px-6 py-2.5 font-bold text-white bg-primary-600 hover:bg-primary-700 rounded-xl shadow-lg shadow-primary-200 transition-colors">Confirm & Deploy</button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {activeTab === 'overview' && (
                    <div className="space-y-12">
                        {/* Stats Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            {[
                                { label: 'Total Candidates', value: stats.users, icon: Users, color: 'bg-blue-500' },
                                { label: 'Exam Blueprints', value: stats.exams, icon: BookOpen, color: 'bg-purple-500' },
                                { label: 'Live Sessions', value: stats.activeSessions, icon: Activity, color: 'bg-emerald-500' }
                            ].map((stat, i) => (
                                <div key={i} className="glass-card p-8 flex items-center space-x-6">
                                    <div className={`h-16 w-16 ${stat.color} rounded-3xl flex items-center justify-center text-white shadow-2xl`}>
                                        <stat.icon className="h-8 w-8" />
                                    </div>
                                    <div>
                                        <p className="text-xs font-black text-gray-400 uppercase tracking-widest">{stat.label}</p>
                                        <p className="text-4xl font-black text-gray-900">{stat.value}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Recent Activity Table (Simulated) */}
                        <div className="glass-card overflow-hidden">
                            <div className="p-8 border-b border-gray-100 flex justify-between items-center bg-white/50">
                                <h2 className="text-xl font-bold text-gray-900">Recent Infrastructure Events</h2>
                                <button className="text-xs font-bold text-primary-600 hover:underline">View Global Logs</button>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-left">
                                    <thead className="bg-gray-50/50 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                                        <tr>
                                            <th className="px-8 py-4">Event Source</th>
                                            <th className="px-8 py-4">Action</th>
                                            <th className="px-8 py-4">Status</th>
                                            <th className="px-8 py-4">Timestamp</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-100 font-medium text-sm text-gray-600">
                                        {[
                                            { source: 'Authentication Service', action: 'Face Embedding Sync', status: 'Success', time: '2 mins ago' },
                                            { source: 'Analysis Engine', action: 'Object Detection Model Load', status: 'Success', time: '15 mins ago' },
                                            { source: 'Exam Server', action: 'New Session Spawn [ID: 882]', status: 'Alert', time: '1 hr ago' },
                                            { source: 'Behavior Engine', action: 'Gaze Tracking Calibrated', status: 'Success', time: '2 hrs ago' },
                                            { source: 'Security Monitor', action: 'Potential Tab Switch Detected [UID: 102]', status: 'Warning', time: '3 hrs ago' }
                                        ].map((row, i) => (
                                            <tr key={i} className="hover:bg-gray-50/50 transition-colors">
                                                <td className="px-8 py-4 text-gray-900">{row.source}</td>
                                                <td className="px-8 py-4">{row.action}</td>
                                                <td className="px-8 py-4">
                                                    <span className={`px-2 py-1 rounded-full text-[10px] uppercase font-black ${row.status === 'Success' ? 'bg-success-100 text-success-700' : row.status === 'Warning' ? 'bg-warning-100 text-warning-700' : 'bg-danger-100 text-danger-700'
                                                        }`}>{row.status}</span>
                                                </td>
                                                <td className="px-8 py-4 text-xs">{row.time}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* User Registry View */}
                {activeTab === 'users' && (
                    <div className="glass-card overflow-hidden">
                        <table className="w-full text-left">
                            <thead className="bg-gray-50/50 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                                <tr>
                                    <th className="px-8 py-4">Candidate</th>
                                    <th className="px-8 py-4">Role</th>
                                    <th className="px-8 py-4">Security Level</th>
                                    <th className="px-8 py-4 text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100 font-medium text-sm">
                                {users.map((user) => (
                                    <tr key={user.id} className="hover:bg-gray-50/50 transition-colors">
                                        <td className="px-8 py-6">
                                            <div className="flex items-center space-x-3">
                                                <div className="h-10 w-10 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center font-black">
                                                    {user.name[0]}
                                                </div>
                                                <div>
                                                    <p className="font-bold text-gray-900">{user.name}</p>
                                                    <p className="text-xs text-gray-500">{user.email}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-8 py-6 uppercase text-[10px] font-black tracking-tighter text-gray-500">{user.role}</td>
                                        <td className="px-8 py-6">
                                            <div className="flex items-center space-x-1">
                                                {[1, 2, 3].map(i => (
                                                    <div key={i} className={`h-1 w-4 rounded-full ${i < 3 ? 'bg-success-500' : 'bg-gray-200'}`}></div>
                                                ))}
                                            </div>
                                        </td>
                                        <td className="px-8 py-6 text-right">
                                            <div className="flex justify-end space-x-2">
                                                <button className="p-2 text-gray-400 hover:text-primary-600 transition-colors"><Edit3 className="h-4 w-4" /></button>
                                                <button className="p-2 text-gray-400 hover:text-danger-600 transition-colors"><Trash2 className="h-4 w-4" /></button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Exam Manager View */}
                {activeTab === 'exams' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {exams.map((exam) => (
                            <div key={exam.id} className="glass-card p-8 group hover:-translate-y-1 transition-all">
                                <div className="flex justify-between items-start mb-6">
                                    <div className="h-12 w-12 bg-primary-50 rounded-2xl flex items-center justify-center text-primary-600 group-hover:bg-primary-600 group-hover:text-white transition-colors">
                                        <BookOpen className="h-6 w-6" />
                                    </div>
                                    <span className="text-xs font-black text-gray-400">{exam.duration} Min Duration</span>
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-2">{exam.title}</h3>
                                <p className="text-sm text-gray-500 leading-relaxed mb-6">{exam.description || 'No description provided.'}</p>
                                <div className="flex justify-between items-center pt-6 border-t border-gray-100">
                                    <div className="flex -space-x-2">
                                        {[1, 2, 3].map(i => (
                                            <div key={i} className="h-8 w-8 rounded-full border-2 border-white bg-gray-200 flex items-center justify-center text-[10px] font-bold">U{i}</div>
                                        ))}
                                    </div>
                                    <button className="flex items-center space-x-2 text-sm font-bold text-primary-600 group-hover:translate-x-1 transition-transform">
                                        <span>Manage Blueprint</span>
                                        <ChevronRight className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
                {activeTab === 'settings' && (
                    <div className="space-y-12 animate-in fade-in duration-700">
                        <section>
                            <h2 className="text-2xl font-black text-gray-900 mb-6 flex items-center gap-2">
                                <Cpu className="h-6 w-6 text-primary-600" /> Infrastructure Health
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                                {Object.entries(health).map(([service, status]) => (
                                    <div key={service} className="glass-card p-6 flex items-center justify-between">
                                        <div>
                                            <p className="text-xs font-black text-gray-400 uppercase">{service} AI</p>
                                            <p className="font-bold text-gray-900 capitalize">{status}</p>
                                        </div>
                                        <div className={`h-3 w-3 rounded-full ${status === 'healthy' ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500 animate-pulse'}`} />
                                    </div>
                                ))}
                            </div>
                        </section>

                        <section>
                            <h2 className="text-2xl font-black text-gray-900 mb-6 flex items-center gap-2">
                                <BarChart3 className="h-6 w-6 text-indigo-600" /> Research Comparative Metrics
                            </h2>
                            {metrics && metrics.single && metrics.multi ? (
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                    {/* Single Modal Card */}
                                    <div className="glass-card p-8 border-l-4 border-amber-500">
                                        <div className="flex justify-between items-start mb-6">
                                            <div>
                                                <h3 className="text-xl font-bold text-gray-900">Single-Modal Vision</h3>
                                                <p className="text-sm text-gray-500">Base AI (Face + Objects only)</p>
                                            </div>
                                            <ShieldAlert className="h-8 w-8 text-amber-500 opacity-20" />
                                        </div>
                                        <div className="space-y-4">
                                            <div className="flex justify-between items-end">
                                                <span className="text-sm font-bold text-gray-600">Accuracy Score</span>
                                                <span className="text-2xl font-black text-amber-600">{(metrics.single.accuracy * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                                                <div className="bg-amber-500 h-full" style={{ width: `${metrics.single.accuracy * 100}%` }} />
                                            </div>
                                            <div className="grid grid-cols-2 gap-4 pt-4">
                                                <div className="bg-gray-50 p-3 rounded-xl">
                                                    <p className="text-[10px] font-black text-gray-400 uppercase">False Alerts</p>
                                                    <p className="text-lg font-bold text-red-600">{metrics.single.false_alert_rate.toFixed(1)}/session</p>
                                                </div>
                                                <div className="bg-gray-50 p-3 rounded-xl">
                                                    <p className="text-[10px] font-black text-gray-400 uppercase">Detection Lag</p>
                                                    <p className="text-lg font-bold text-gray-700">{metrics.single.avg_latency.toFixed(2)}s</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Multimodal Card */}
                                    <div className="glass-card p-8 border-l-4 border-green-500 bg-green-50/10">
                                        <div className="flex justify-between items-start mb-6">
                                            <div>
                                                <h3 className="text-xl font-bold text-gray-900">Multimodal Fusion</h3>
                                                <p className="text-sm text-gray-500">Full AI Stack (Vision + Audio + Behavioral)</p>
                                            </div>
                                            <Trophy className="h-8 w-8 text-green-500 opacity-20" />
                                        </div>
                                        <div className="space-y-4">
                                            <div className="flex justify-between items-end">
                                                <span className="text-sm font-bold text-gray-600">Accuracy Score</span>
                                                <span className="text-2xl font-black text-green-600">{(metrics.multi.accuracy * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                                                <div className="bg-green-500 h-full shadow-[0_0_15px_rgba(34,197,94,0.3)]" style={{ width: `${metrics.multi.accuracy * 100}%` }} />
                                            </div>
                                            <div className="grid grid-cols-2 gap-4 pt-4">
                                                <div className="bg-white p-3 rounded-xl border border-green-100">
                                                    <p className="text-[10px] font-black text-gray-400 uppercase">False Alerts</p>
                                                    <p className="text-lg font-bold text-green-600">{metrics.multi.false_alert_rate.toFixed(1)}/session</p>
                                                </div>
                                                <div className="bg-white p-3 rounded-xl border border-green-100">
                                                    <p className="text-[10px] font-black text-gray-400 uppercase">Detection Lag</p>
                                                    <p className="text-lg font-bold text-gray-700">{metrics.multi.avg_latency.toFixed(2)}s</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="glass-card p-12 text-center text-gray-400">
                                    <Monitor className="h-12 w-12 mx-auto mb-4 opacity-20" />
                                    <p className="font-bold">No research data available.</p>
                                    <p className="text-sm mt-1">Run `python -m experiments.experiment_runner` to generate metrics.</p>
                                </div>
                            )}
                        </section>
                    </div>
                )}
            </main>
        </div>
    );
};

export default AdminDashboard;
