import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import Login from './pages/Login';
import Register from './pages/Register';
import StudentDashboard from './pages/StudentDashboard';
import SupervisorDashboard from './pages/SupervisorDashboard';
import EnhancedExamSession from './components/EnhancedExamSession';
import AdminDashboard from './pages/AdminDashboard';
import ReportCenter from './pages/ReportCenter';
import LiveMonitoring from './pages/LiveMonitoring';
import Layout from './components/Layout';

function ProtectedRoute({ children, allowedRoles = [] }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles.length > 0 && !allowedRoles.includes(user.role)) {
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
}

function AppRoutes() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={!user ? <Login /> : <Navigate to="/dashboard" replace />} />
      <Route path="/register" element={!user ? <Register /> : <Navigate to="/dashboard" replace />} />

      <Route path="/dashboard" element={
        <ProtectedRoute>
          <Layout>
            {user?.role === 'student' ? <StudentDashboard /> : <SupervisorDashboard />}
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/exam/:sessionId" element={
        <ProtectedRoute allowedRoles={['student']}>
          <EnhancedExamSession />
        </ProtectedRoute>
      } />

      <Route path="/supervisor" element={
        <ProtectedRoute allowedRoles={['supervisor', 'admin']}>
          <Layout>
            <SupervisorDashboard />
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/admin" element={
        <ProtectedRoute allowedRoles={['admin']}>
          <AdminDashboard />
        </ProtectedRoute>
      } />

      <Route path="/report/:sessionId" element={
        <ProtectedRoute allowedRoles={['supervisor', 'admin']}>
          <ReportCenter />
        </ProtectedRoute>
      } />

      <Route path="/live/:sessionId" element={
        <ProtectedRoute allowedRoles={['supervisor', 'admin']}>
          <Layout>
            <LiveMonitoring />
          </Layout>
        </ProtectedRoute>
      } />

      <Route path="/unauthorized" element={
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 mb-4">Unauthorized</h1>
            <p className="text-gray-600">You don't have permission to access this page.</p>
          </div>
        </div>
      } />

      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 mb-4">Page Not Found</h1>
            <p className="text-gray-600">The page you're looking for doesn't exist.</p>
          </div>
        </div>
      } />
    </Routes>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <SocketProvider>
          <div className="App">
            <AppRoutes />
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
                success: {
                  duration: 3000,
                  iconTheme: {
                    primary: '#22c55e',
                    secondary: '#fff',
                  },
                },
                error: {
                  duration: 5000,
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#fff',
                  },
                },
              }}
            />
          </div>
        </SocketProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;