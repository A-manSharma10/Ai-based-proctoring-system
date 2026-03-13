import React, { useEffect, useState } from 'react';
import { AlertTriangle, X, Eye, Phone, Users, Volume2, Activity } from 'lucide-react';

const ViolationPopup = ({ violation, onClose, warningsLeft }) => {
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    const duration = 8000; // 8 seconds
    const interval = 50;
    const decrement = (interval / duration) * 100;

    const timer = setInterval(() => {
      setProgress(prev => {
        if (prev <= 0) {
          clearInterval(timer);
          onClose();
          return 0;
        }
        return prev - decrement;
      });
    }, interval);

    return () => clearInterval(timer);
  }, [onClose]);

  const getIcon = () => {
    switch (violation.type) {
      case 'face':
      case 'no_face':
      case 'multiple_faces':
        return <Users className="h-6 w-6" />;
      case 'gaze':
      case 'looking_away':
        return <Eye className="h-6 w-6" />;
      case 'object':
      case 'phone':
      case 'book':
        return <Phone className="h-6 w-6" />;
      case 'audio':
      case 'speech':
        return <Volume2 className="h-6 w-6" />;
      default:
        return <AlertTriangle className="h-6 w-6" />;
    }
  };

  const getSeverityColor = () => {
    switch (violation.severity) {
      case 'critical':
        return 'bg-red-600 border-red-700';
      case 'high':
        return 'bg-orange-600 border-orange-700';
      case 'medium':
        return 'bg-yellow-600 border-yellow-700';
      default:
        return 'bg-blue-600 border-blue-700';
    }
  };

  return (
    <div className="fixed top-20 right-6 z-50 animate-slide-in-right">
      <div className={`${getSeverityColor()} text-white rounded-2xl shadow-2xl border-2 overflow-hidden max-w-md`}>
        {/* Progress bar */}
        <div className="h-1 bg-white/20">
          <div 
            className="h-full bg-white transition-all duration-50 ease-linear"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Content */}
        <div className="p-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              {getIcon()}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="font-bold text-lg leading-tight">
                    {violation.title}
                  </h3>
                  <p className="text-sm opacity-90 mt-1">
                    {violation.message}
                  </p>
                  {violation.confidence && (
                    <p className="text-xs opacity-75 mt-1">
                      Confidence: {(violation.confidence * 100).toFixed(0)}%
                    </p>
                  )}
                </div>
                
                <button
                  onClick={onClose}
                  className="flex-shrink-0 hover:bg-white/20 rounded-lg p-1 transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {/* Warnings left */}
              {warningsLeft !== undefined && (
                <div className="mt-3 pt-3 border-t border-white/20">
                  <div className="flex items-center justify-between text-sm">
                    <span className="opacity-90">Warnings Remaining:</span>
                    <span className="font-bold text-lg">{warningsLeft}</span>
                  </div>
                  {warningsLeft <= 2 && (
                    <p className="text-xs opacity-75 mt-1">
                      ⚠️ Further violations may result in exam termination
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ViolationPopup;
