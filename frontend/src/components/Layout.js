import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useSocket } from '../contexts/SocketContext';
import { LogOut, User, Wifi, WifiOff, Bell } from 'lucide-react';

const Layout = ({ children }) => {
  const { user, logout } = useAuth();
  const { connected, alerts } = useSocket();

  const unreadAlerts = alerts.filter(alert => !alert.acknowledged).length;

  return (
    <div className="min-h-screen mesh-gradient">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-white/40 shadow-premium">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo and Title */}
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-xl font-bold text-gray-900">
                  AI Exam Proctoring
                </h1>
              </div>
            </div>

            {/* Status and User Info */}
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                {connected ? (
                  <div className="flex items-center text-success-600">
                    <Wifi className="h-4 w-4" />
                    <span className="text-sm font-medium">Connected</span>
                  </div>
                ) : (
                  <div className="flex items-center text-danger-600">
                    <WifiOff className="h-4 w-4" />
                    <span className="text-sm font-medium">Disconnected</span>
                  </div>
                )}
              </div>

              {/* Alerts (for supervisors) */}
              {(user?.role === 'supervisor' || user?.role === 'admin') && (
                <div className="relative">
                  <button className="relative p-2 text-gray-600 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-md">
                    <Bell className="h-5 w-5" />
                    {unreadAlerts > 0 && (
                      <span className="absolute -top-1 -right-1 h-4 w-4 bg-danger-500 text-white text-xs rounded-full flex items-center justify-center">
                        {unreadAlerts > 9 ? '9+' : unreadAlerts}
                      </span>
                    )}
                  </button>
                </div>
              )}

              {/* User Menu */}
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <User className="h-5 w-5 text-gray-600" />
                  <div className="text-sm">
                    <div className="font-medium text-gray-900">{user?.name}</div>
                    <div className="text-gray-500 capitalize">{user?.role}</div>
                  </div>
                </div>

                <button
                  onClick={logout}
                  className="flex items-center space-x-1 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white/40 backdrop-blur-sm border-t border-white/20 mt-auto">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center text-sm text-gray-500 space-y-4 md:space-y-0">
            <div className="font-semibold text-gray-600">AI Exam Proctoring</div>
            <p>&copy; 2024 Secure Examination Protocol. Built for Integrity.</p>
            <div className="flex space-x-6">
              <span className="hover:text-primary-600 cursor-pointer transition-colors">Privacy</span>
              <span className="hover:text-primary-600 cursor-pointer transition-colors">Terms</span>
              <span className="hover:text-primary-600 cursor-pointer transition-colors">Support</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;