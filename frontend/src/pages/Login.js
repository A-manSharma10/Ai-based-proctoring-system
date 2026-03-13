import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Camera, Eye, EyeOff } from 'lucide-react';
import Webcam from 'react-webcam';

const Login = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [faceImage, setFaceImage] = useState(null);
  const webcamRef = useRef(null);
  const { login } = useAuth();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const capturePhoto = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setFaceImage(imageSrc);
    setShowCamera(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await login(formData.email, formData.password, faceImage);
      if (!result.success) {
        // Error is already handled in the login function
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center mesh-gradient py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 bg-primary-600 rounded-2xl flex items-center justify-center shadow-lg shadow-primary-500/30 mb-6">
            <Camera className="h-8 w-8 text-white" />
          </div>
          <h2 className="text-4xl font-extrabold text-gray-900 tracking-tight">
            Welcome Back
          </h2>
          <p className="mt-2 text-sm text-gray-500 font-medium">
            Secure AI proctoring for institutional excellence
          </p>
        </div>

        <div className="glass-card p-8 space-y-6">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-semibold text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="input"
                  placeholder="name@institution.edu"
                  value={formData.email}
                  onChange={handleChange}
                />
              </div>
              <div className="relative">
                <label htmlFor="password" className="block text-sm font-semibold text-gray-700 mb-1">
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    required
                    className="input pr-12"
                    placeholder="••••••••"
                    value={formData.password}
                    onChange={handleChange}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-primary-500 transition-colors"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Face Recognition Section */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-semibold text-gray-700">
                  Biometric Auth
                </label>
                <button
                  type="button"
                  onClick={() => setShowCamera(!showCamera)}
                  className="flex items-center space-x-2 text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors"
                >
                  <Camera className="h-4 w-4" />
                  <span>{showCamera ? 'Hide Camera' : 'Face Scan'}</span>
                </button>
              </div>

              {showCamera && (
                <div className="space-y-4 animate-in zoom-in-95 duration-300">
                  <div className="relative rounded-2xl overflow-hidden border-2 border-primary-100 shadow-inner">
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      className="w-full"
                      videoConstraints={{
                        width: 640,
                        height: 480,
                        facingMode: 'user'
                      }}
                    />
                    <div className="absolute inset-0 border-[20px] border-black/10 pointer-events-none"></div>
                  </div>
                  <button
                    type="button"
                    onClick={capturePhoto}
                    className="w-full btn btn-primary py-3"
                  >
                    Capture Biometrics
                  </button>
                </div>
              )}

              {faceImage && !showCamera && (
                <div className="flex items-center space-x-4 p-3 bg-primary-50 rounded-xl border border-primary-100 animate-in slide-in-from-top-2">
                  <img
                    src={faceImage}
                    alt="Captured face"
                    className="w-16 h-12 object-cover rounded-lg shadow-sm"
                  />
                  <div className="flex-1">
                    <p className="text-xs font-bold text-primary-700">Face ID Ready</p>
                    <p className="text-[10px] text-primary-500">Biometric token generated</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFaceImage(null)}
                    className="p-2 text-danger-500 hover:bg-danger-50 rounded-lg transition-colors"
                  >
                    <EyeOff className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn btn-primary py-3 text-lg font-bold shadow-xl shadow-primary-500/20"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="loading-spinner border-white/30 border-t-white mr-3"></div>
                  Authenticating...
                </div>
              ) : (
                'Sign In'
              )}
            </button>

            <div className="text-center pt-4 border-t border-gray-100">
              <p className="text-sm text-gray-500">
                New to the platform?{' '}
                <Link
                  to="/register"
                  className="font-bold text-primary-600 hover:text-primary-700 underline-offset-4 hover:underline"
                >
                  Create an account
                </Link>
              </p>
            </div>
          </form>
        </div>

        {/* Demo Credentials - Professional Style */}
        <div className="bg-white/40 backdrop-blur-sm rounded-2xl p-6 border border-white/40 shadow-sm animate-in fade-in slide-in-from-top-4 delay-300 duration-1000">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4">Demo Access</h3>
          <div className="grid grid-cols-1 gap-3">
            {[
              { role: 'Student', email: 'student1@exam.com' },
              { role: 'Supervisor', email: 'supervisor@exam.com' },
              { role: 'Admin', email: 'admin@exam.com' }
            ].map((cred) => (
              <div key={cred.role} className="flex items-center justify-between text-sm group cursor-pointer" onClick={() => setFormData({ email: cred.email, password: 'password' })}>
                <span className="text-gray-600 font-medium group-hover:text-primary-600 transition-colors">{cred.role}</span>
                <span className="text-gray-400 font-mono text-xs opacity-60 group-hover:opacity-100">{cred.email}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;