import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
  Camera as LucideCamera,
  Eye as LucideEye,
  EyeOff as LucideEyeOff,
  XCircle as LucideXCircle
} from 'lucide-react';
import Webcam from 'react-webcam';
import toast from 'react-hot-toast';

const Register = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'student'
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [faceImage, setFaceImage] = useState(null);
  const webcamRef = useRef(null);
  const { register } = useAuth();

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

    if (formData.password !== formData.confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }

    if (formData.password.length < 6) {
      toast.error('Password must be at least 6 characters long');
      return;
    }

    setLoading(true);

    try {
      const userData = {
        name: formData.name,
        email: formData.email,
        password: formData.password,
        role: formData.role,
        faceImage: faceImage
      };

      const result = await register(userData);
      if (result.success) {
        // Registration successful, user will be redirected to login
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center mesh-gradient py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-xl w-full space-y-8 animate-in mt-8 fade-in slide-in-from-bottom-4 duration-700">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 bg-accent-600 rounded-2xl flex items-center justify-center shadow-lg shadow-accent-500/30 mb-6 font-bold text-white text-2xl">
            AI
          </div>
          <h2 className="text-4xl font-extrabold text-gray-900 tracking-tight">
            Create Account
          </h2>
          <p className="mt-2 text-sm text-gray-500 font-medium">
            Join the next generation of academic integrity
          </p>
        </div>

        <div className="glass-card p-10 space-y-8">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4 col-span-2">
                <div>
                  <label htmlFor="name" className="block text-sm font-semibold text-gray-700 mb-1">
                    Full Name
                  </label>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    required
                    className="input"
                    placeholder="John Doe"
                    value={formData.name}
                    onChange={handleChange}
                  />
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-semibold text-gray-700 mb-1">
                    Institution Email
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    className="input"
                    placeholder="john@university.edu"
                    value={formData.email}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="role" className="block text-sm font-semibold text-gray-700 mb-1">
                  Account Type
                </label>
                <select
                  id="role"
                  name="role"
                  className="input appearance-none bg-no-repeat bg-right"
                  style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' fill=\'none\' viewBox=\'0 0 24 24\' stroke=\'%236B7280\'%3E%3Cpath stroke-linecap=\'round\' stroke-linejoin=\'round\' stroke-width=\'2\' d=\'M19 9l-7 7-7-7\'%3E%3C/path%3E%3C/svg%3E")', backgroundSize: '1.5em' }}
                  value={formData.role}
                  onChange={handleChange}
                >
                  <option value="student">Student</option>
                  <option value="supervisor">Supervisor</option>
                </select>
              </div>

              <div className="md:col-span-1"></div>

              <div className="relative">
                <label htmlFor="password" className="block text-sm font-semibold text-gray-700 mb-1">
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    className="input pr-12"
                    placeholder="••••••••"
                    value={formData.password}
                    onChange={handleChange}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <LucideEyeOff className="h-5 w-5" /> : <LucideEye className="h-5 w-5" />}
                  </button>
                </div>
              </div>

              <div className="relative">
                <label htmlFor="confirmPassword" className="block text-sm font-semibold text-gray-700 mb-1">
                  Confirm
                </label>
                <div className="relative">
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    className="input pr-12"
                    placeholder="••••••••"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? <LucideEyeOff className="h-5 w-5" /> : <LucideEye className="h-5 w-5" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Face Registration Section */}
            <div className="space-y-4 pt-4 border-t border-white/20">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-semibold text-gray-700">
                    Biometric Enrollment
                  </label>
                  <p className="text-[10px] text-gray-400 font-medium">Verified identity for exam access</p>
                </div>
                <button
                  type="button"
                  onClick={() => setShowCamera(!showCamera)}
                  className="flex items-center space-x-2 text-sm font-bold text-accent-600 hover:text-accent-700"
                >
                  <LucideCamera className="h-4 w-4" />
                  <span>{showCamera ? 'Dismiss' : 'Start Scan'}</span>
                </button>
              </div>

              {showCamera && (
                <div className="space-y-4 animate-in zoom-in-95 duration-300">
                  <div className="relative rounded-2xl overflow-hidden border-2 border-accent-100 shadow-2xl">
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
                  </div>
                  <button
                    type="button"
                    onClick={capturePhoto}
                    className="w-full btn bg-accent-600 text-white hover:bg-accent-700 py-3 shadow-lg shadow-accent-500/20"
                  >
                    Capture Biometrics
                  </button>
                </div>
              )}

              {faceImage && !showCamera && (
                <div className="group relative w-32 h-24 rounded-2xl overflow-hidden border-2 border-accent-200 shadow-lg animate-in slide-in-from-left-4">
                  <img src={faceImage} alt="Captured face" className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => setFaceImage(null)}
                    className="absolute top-1 right-1 p-1 bg-danger-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <LucideXCircle className="h-3 w-3" />
                  </button>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn btn-primary py-4 text-lg font-bold shadow-xl shadow-primary-500/30"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="loading-spinner border-white/30 border-t-white mr-3"></div>
                  Registering...
                </div>
              ) : (
                'Create Secure Account'
              )}
            </button>

            <div className="text-center pt-4 border-t border-white/20">
              <p className="text-sm text-gray-500">
                Member of an institution?{' '}
                <Link
                  to="/login"
                  className="font-bold text-primary-600 hover:text-primary-700"
                >
                  Sign in
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Register;