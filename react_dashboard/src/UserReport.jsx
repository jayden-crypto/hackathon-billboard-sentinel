import React, { useState, useRef, useEffect } from 'react';
import './UserReport.css';

export default function UserReport() {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isLoadingCamera, setIsLoadingCamera] = useState(false);
  const [reportData, setReportData] = useState({
    description: '',
    location: '',
    urgency: 'medium',
    consent: false
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [cameraInitialized, setCameraInitialized] = useState(false);
  const [photoCaptured, setPhotoCaptured] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  // Simple camera check when component mounts
  useEffect(() => {
    console.log('🚀 Component mounted, checking camera support...');
    
    // Simple check for camera support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Camera not supported in this browser');
      return;
    }
    
    // Set camera as ready for initialization
    setCameraInitialized(true);
    console.log('✅ Camera support detected, ready for user interaction');
  }, []);

  // Auto-retry camera initialization if it fails
  useEffect(() => {
    if (cameraError && !cameraInitialized) {
      const retryTimer = setTimeout(() => {
        console.log('🔄 Auto-retrying camera initialization...');
        // Clear error and try to initialize again
        setCameraError('');
        setCameraInitialized(false);
      }, 3000); // Retry after 3 seconds

      return () => clearTimeout(retryTimer);
    }
  }, [cameraError, cameraInitialized]);

  // Listen for camera permission changes
  useEffect(() => {
    const handlePermissionChange = async () => {
      try {
        const permissions = await navigator.permissions.query({ name: 'camera' });
        if (permissions.state === 'granted' && !cameraInitialized) {
          console.log('🎥 Camera permission granted, initializing...');
          await openCamera(true);
        }
      } catch (error) {
        console.log('Could not check permission changes');
      }
    };

    // Try to set up permission change listener
    try {
      navigator.permissions.query({ name: 'camera' }).then(permissions => {
        permissions.addEventListener('change', handlePermissionChange);
        return () => permissions.removeEventListener('change', handlePermissionChange);
      });
    } catch (error) {
      console.log('Permission change listener not supported');
    }
  }, [cameraInitialized]);

  const openCamera = async (silent = false) => {
    try {
      setCameraError('');
      setIsLoadingCamera(true);
      console.log('=== STARTING CAMERA INITIALIZATION ===');
      
      // Check if camera is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported in this browser');
      }

      // Check if we already have a working stream
      if (streamRef.current && streamRef.current.active) {
        console.log('✅ Camera stream already active, reusing...');
        setIsCameraOpen(true);
        setIsLoadingCamera(false);
        setCameraInitialized(true);
        return;
      }

      // Progressive fallback constraints - try from most specific to most basic
      const constraintOptions = [
        // Try back camera first
        { video: { facingMode: 'environment' } },
        // Try front camera
        { video: { facingMode: 'user' } },
        // Try any camera with basic constraints
        { video: { width: { ideal: 640 }, height: { ideal: 480 } } },
        // Most basic - just video
        { video: true }
      ];

      let stream = null;
      let lastError = null;

      for (let i = 0; i < constraintOptions.length; i++) {
        try {
          console.log(`Trying camera constraints ${i + 1}/${constraintOptions.length}:`, constraintOptions[i]);
          stream = await navigator.mediaDevices.getUserMedia(constraintOptions[i]);
          console.log('✅ Camera access granted with constraints:', constraintOptions[i]);
          break;
        } catch (error) {
          console.log(`❌ Constraints ${i + 1} failed:`, error.message);
          lastError = error;
          if (i === constraintOptions.length - 1) {
            throw error;
          }
        }
      }

      if (!stream) {
        throw lastError || new Error('Failed to get camera stream');
      }
      
      streamRef.current = stream;
      
      // Show camera interface immediately
      setIsCameraOpen(true);
      setIsLoadingCamera(false);
      setCameraInitialized(true);
      setCameraError('');
      
      if (videoRef.current) {
        const video = videoRef.current;
        console.log('Setting up video element...');
        
        // Wait for camera interface to render, then set up video
        setTimeout(async () => {
          try {
            // Clear any existing source
            if (video.srcObject) {
              const oldTracks = video.srcObject.getTracks();
              oldTracks.forEach(track => track.stop());
              video.srcObject = null;
            }
            
            // Set the stream
            video.srcObject = stream;
            
            // Set video properties for better compatibility
            video.setAttribute('playsinline', '');
            video.setAttribute('webkit-playsinline', '');
            video.muted = true;
            video.autoplay = true;
            
            // Force video dimensions
            video.style.width = '100%';
            video.style.height = 'auto';
            video.style.objectFit = 'cover';
            
            // Try to play the video
            await video.play();
            console.log('✅ Video playing successfully');
            
            // Double check video is actually showing content
            setTimeout(() => {
              if (video.videoWidth === 0 || video.videoHeight === 0) {
                console.log('⚠️ Video dimensions are 0, trying alternative approach...');
                
                // Try creating a new video element
                const newVideo = document.createElement('video');
                newVideo.srcObject = stream;
                newVideo.setAttribute('playsinline', '');
                newVideo.setAttribute('webkit-playsinline', '');
                newVideo.muted = true;
                newVideo.autoplay = true;
                newVideo.style.width = '100%';
                newVideo.style.height = 'auto';
                newVideo.style.objectFit = 'cover';
                
                // Replace the existing video
                if (video.parentNode) {
                  video.parentNode.replaceChild(newVideo, video);
                  videoRef.current = newVideo;
                  newVideo.play().catch(e => console.log('New video play failed:', e));
                }
              }
            }, 1000);
            
          } catch (playError) {
            console.error('Video play failed:', playError);
            // Try alternative approach
            try {
              video.load();
              await video.play();
              console.log('✅ Video playing after reload');
            } catch (retryError) {
              console.error('Video retry failed:', retryError);
              // Still show interface, user can try capture
            }
          }
        }, 200);
        
      } else {
        console.log('❌ Video element not found - but camera interface is shown');
      }
    } catch (error) {
      console.error('❌ Camera error:', error);
      setIsLoadingCamera(false);
      
      if (error.name === 'NotAllowedError') {
        setCameraError('Camera permission denied. Please allow camera access and try again.');
      } else if (error.name === 'NotFoundError') {
        setCameraError('No camera found on this device.');
      } else if (error.name === 'NotReadableError') {
        setCameraError('Camera is already in use by another application.');
      } else {
        setCameraError(`Camera error: ${error.message}`);
      }
    }
  };

  const closeCamera = () => {
    console.log('Closing camera...');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        console.log('Stopping track:', track.kind);
        track.stop();
      });
      streamRef.current = null;
    }
    setIsCameraOpen(false);
    // Don't clear capturedImage here - only clear it when retaking photo
    setCameraError('');
  };

  const capturePhoto = () => {
    console.log('Capturing photo...');
    if (videoRef.current) {
      // Check if video is actually ready
      const video = videoRef.current;
      console.log('Video state:', {
        readyState: video.readyState,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        paused: video.paused,
        ended: video.ended
      });
      
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          
          // Clear canvas first
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          // Draw the video frame to canvas with proper orientation
          // This ensures the captured image is not mirrored
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Convert to data URL for localStorage compatibility
          const dataUrl = canvas.toDataURL('image/jpeg', 0.7); // Compressed for storage
          console.log('Photo captured successfully!');
          console.log('Data URL length:', dataUrl.length);
          console.log('Image dimensions:', canvas.width, 'x', canvas.height);
          
          // Also create blob for form submission
          canvas.toBlob((blob) => {
            if (blob) {
              const blobUrl = URL.createObjectURL(blob);
              
              // Set both data URL and blob
              setCapturedImage({ 
                blob, 
                url: blobUrl,
                dataUrl: dataUrl 
              });
              
              // Close camera after setting the image
              setTimeout(() => {
                closeCamera();
              }, 100);
            } else {
              console.error('Failed to create blob from canvas');
              alert('Failed to capture photo. Please try again.');
            }
          }, 'image/jpeg', 0.7);
        } catch (error) {
          console.error('Error capturing photo:', error);
          alert('Error capturing photo. Please try again.');
        }
      } else {
        console.error('Video not ready for capture. Ready state:', video.readyState);
        alert('Camera not ready. Please wait a moment and try again.');
      }
    } else {
      alert('Camera not available. Please try again.');
    }
  };

  const reverseGeocode = async (lat, lon) => {
    try {
      // Use OpenStreetMap Nominatim API (free, no API key required)
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=16&addressdetails=1`
      );
      const data = await response.json();
      
      if (data && data.display_name) {
        // Extract meaningful location parts
        const address = data.address || {};
        const parts = [];
        
        if (address.road) parts.push(address.road);
        if (address.neighbourhood) parts.push(address.neighbourhood);
        if (address.suburb) parts.push(address.suburb);
        if (address.city || address.town || address.village) {
          parts.push(address.city || address.town || address.village);
        }
        
        return parts.length > 0 ? parts.join(', ') : data.display_name.split(',').slice(0, 3).join(',');
      }
    } catch (error) {
      console.error('Reverse geocoding failed:', error);
    }
    return `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          const locationName = await reverseGeocode(lat, lon);
          
          setReportData(prev => ({
            ...prev,
            location: locationName
          }));
        },
        (error) => {
          console.error('Error getting location:', error);
          alert('Unable to get your location. Please enter manually.');
        }
      );
    } else {
      alert('Geolocation is not supported by this browser.');
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setReportData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  // Function to clear old localStorage data to prevent quota issues
  const clearOldStorageData = () => {
    try {
      // Clear old image data (keep only last 5 images)
      const reports = JSON.parse(localStorage.getItem('githubPageReports') || '[]');
      const reportsToKeep = reports.slice(0, 5);
      const imagesToKeep = reportsToKeep.map(r => r.image.replace('placeholder-', '')).filter(Boolean);
      
      // Remove old image keys
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith('report-image-')) {
          const reportId = key.replace('report-image-', '');
          if (!imagesToKeep.includes(reportId)) {
            localStorage.removeItem(key);
          }
        }
      });
      
      console.log('Cleared old storage data');
    } catch (e) {
      console.log('Error clearing old storage:', e);
    }
  };

  const submitReport = async () => {
    if (!capturedImage || !reportData.consent) {
      alert('Please take a photo and give consent to submit a report.');
      return;
    }

    setIsSubmitting(true);
    
    // Clear old data before storing new report
    clearOldStorageData();
    
    try {
      // Check if we're on GitHub Pages or localhost
      const isGitHubPages = window.location.hostname.includes('github.io');
      
      if (isGitHubPages) {
        // Parse coordinates from location if available
        let lat = 30.354 + Math.random() * 0.01;
        let lon = 76.366 + Math.random() * 0.01;
        let locationName = reportData.location;
        
        if (reportData.location && reportData.location.includes(',')) {
          const coords = reportData.location.split(',').map(c => parseFloat(c.trim()));
          if (coords.length === 2 && !isNaN(coords[0]) && !isNaN(coords[1])) {
            lat = coords[0];
            lon = coords[1];
            // Try to get readable name for coordinates
            try {
              locationName = await reverseGeocode(lat, lon);
            } catch (e) {
              locationName = reportData.location;
            }
          }
        }
        
        // Store report in localStorage for GitHub Pages (without large image data)
        const reportId = Date.now().toString();
        const newReport = {
          id: reportId,
          location: locationName || `Location ${reportId.slice(-8)}`,
          coordinates: [lat, lon],
          status: "Pending Review",
          timestamp: new Date().toISOString(),
          detections: ["Billboard", "No License"],
          image: `placeholder-${reportId}`, // Use placeholder instead of full image data
          description: reportData.description,
          archived: "false",
          archived_at: null
        };
        
        try {
          // Get existing reports from localStorage
          const existingReports = JSON.parse(localStorage.getItem('githubPageReports') || '[]');
          
          // Limit to last 10 reports to prevent storage overflow
          const limitedReports = existingReports.slice(0, 9);
          limitedReports.unshift(newReport); // Add to beginning
          
          localStorage.setItem('githubPageReports', JSON.stringify(limitedReports));
          
          // Store image separately with size limit check
          const imageKey = `report-image-${reportId}`;
          try {
            // Compress image data if too large
            let imageData = capturedImage.dataUrl;
            if (imageData && imageData.length > 500000) { // 500KB limit
              // Create a smaller version
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
              const img = new Image();
              img.onload = () => {
                // Resize to max 400x300
                const maxWidth = 400;
                const maxHeight = 300;
                let { width, height } = img;
                
                if (width > maxWidth || height > maxHeight) {
                  const ratio = Math.min(maxWidth / width, maxHeight / height);
                  width *= ratio;
                  height *= ratio;
                }
                
                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                
                const compressedData = canvas.toDataURL('image/jpeg', 0.6);
                try {
                  localStorage.setItem(imageKey, compressedData);
                } catch (e) {
                  console.log('Could not store compressed image, using placeholder');
                }
              };
              img.src = imageData;
            } else if (imageData) {
              localStorage.setItem(imageKey, imageData);
            }
          } catch (imageError) {
            console.log('Could not store image data:', imageError.message);
          }
        } catch (storageError) {
          console.error('localStorage error:', storageError);
          alert('Storage limit reached. Please clear some data or use the local API version.');
          return;
        }
        
        console.log('Report stored in localStorage:', newReport);
        console.log('Image data URL length:', newReport.image ? newReport.image.length : 'No image');
        
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setShowSuccess(true);
        setPhotoCaptured(true);
        
        // Notify parent to refresh dashboard
        if (window.refreshDashboard) {
          window.refreshDashboard();
        }
        
      } else {
        // Real API call for localhost - format data for backend API
        const formData = new FormData();
        formData.append('image', capturedImage.blob, 'billboard-detection.jpg');
        
        // Parse location coordinates or use default
        let lat = 30.354, lon = 76.366; // Default coordinates
        if (reportData.location && reportData.location.includes(',')) {
          const coords = reportData.location.split(',').map(c => parseFloat(c.trim()));
          if (coords.length === 2 && !isNaN(coords[0]) && !isNaN(coords[1])) {
            lat = coords[0];
            lon = coords[1];
          }
        }
        
        formData.append('lat', lat.toString());
        formData.append('lon', lon.toString());
        formData.append('device_heading', '0.0');
        
        // Create mock detection data for the backend
        const detections = [{
          bbox: [0.1, 0.1, 0.9, 0.9],
          corners: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
          est_width_m: 10.0,
          est_height_m: 3.0,
          confidence: 0.85,
          qr_text: '',
          ocr_text: reportData.description || 'Billboard detected',
          license_id: ''
        }];
        
        formData.append('detections_json', JSON.stringify(detections));
        
        const response = await fetch('http://localhost:8000/api/reports', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          console.log('Report submitted successfully:', result);
          setShowSuccess(true);
          setPhotoCaptured(true);
          
          // Notify parent component to refresh dashboard
          if (window.refreshDashboard) {
            window.refreshDashboard();
          }
        } else {
          const errorText = await response.text();
          throw new Error(`Backend error: ${errorText}`);
        }
      }
      
      // Reset form on success
      setCapturedImage(null);
      setReportData({
        description: '',
        location: '',
        urgency: 'medium',
        consent: false
      });
      
      // Hide success message after 5 seconds
      setTimeout(() => {
        setShowSuccess(false);
        setPhotoCaptured(false);
      }, 5000);
      
    } catch (error) {
      console.error('Error submitting report:', error);
      alert(`Failed to submit report: ${error.message}. Please try again.`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    openCamera();
  };

  // Debug function to check camera state
  const debugCamera = () => {
    console.log('=== CAMERA DEBUG INFO ===');
    console.log('isCameraOpen:', isCameraOpen);
    console.log('isLoadingCamera:', isLoadingCamera);
    console.log('cameraError:', cameraError);
    console.log('capturedImage:', capturedImage);
    
    if (videoRef.current) {
      const video = videoRef.current;
      const debugInfo = {
        readyState: video.readyState,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        paused: video.paused,
        ended: video.ended,
        srcObject: video.srcObject ? 'Set' : 'Not set',
        currentSrc: video.currentSrc,
        muted: video.muted,
        autoplay: video.autoplay,
        playsInline: video.playsInline
      };
      console.log('Video element state:', debugInfo);
      
      // Show debug info to user
      alert(`Camera Debug Info:
Video Ready State: ${debugInfo.readyState} (4=loaded)
Video Dimensions: ${debugInfo.videoWidth}x${debugInfo.videoHeight}
Stream: ${debugInfo.srcObject}
Paused: ${debugInfo.paused}
Muted: ${debugInfo.muted}
PlaysInline: ${debugInfo.playsInline}`);
    } else {
      console.log('Video element not found');
      alert('Video element not found');
    }
    
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks().map(t => ({
        kind: t.kind,
        enabled: t.enabled,
        readyState: t.readyState,
        label: t.label
      }));
      console.log('Stream tracks:', tracks);
      alert(`Stream tracks: ${tracks.length} found\n${tracks.map(t => `${t.kind}: ${t.readyState} (${t.label})`).join('\n')}`);
    } else {
      console.log('No stream reference');
      alert('No stream reference found');
      if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];
        console.log('File selected:', file.name, file.size, 'bytes');
        
        // Create both blob URL and data URL
        const imageUrl = URL.createObjectURL(file);
        
        // Convert file to data URL for localStorage
        const reader = new FileReader();
        reader.onload = (e) => {
          const dataUrl = e.target.result;
          setCapturedImage({ 
            blob: file, 
            url: imageUrl,
            dataUrl: dataUrl 
          });
          setPhotoCaptured(true);
        };
        reader.readAsDataURL(file);
      }
      console.log('Photo uploaded via file input');
    }
  };

  // Main photo button handler - use file input directly (most reliable on mobile)
  const handleTakePhoto = () => {
    console.log('=== TAKE PHOTO BUTTON PRESSED ===');
    
    // Directly trigger file input with camera capture
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Simple camera function that bypasses React complexity
  const trySimpleCamera = async () => {
    try {
      console.log('=== TRYING SIMPLE CAMERA ===');
      
      // Create a simple video element directly
      const video = document.createElement('video');
      video.style.width = '100%';
      video.style.height = 'auto';
      video.style.backgroundColor = '#000';
      video.setAttribute('playsinline', '');
      video.setAttribute('autoplay', '');
      video.muted = true;
      
      // Get basic video stream
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      
      // Replace existing video or add to container
      const videoContainer = document.querySelector('.camera-video');
      if (videoContainer && videoContainer.parentNode) {
        videoContainer.parentNode.replaceChild(video, videoContainer);
        videoRef.current = video;
      }
      
      await video.play();
      console.log('✅ Simple camera working');
      
      // Update state
      streamRef.current = stream;
      setIsCameraOpen(true);
      setCameraError('');
      
    } catch (error) {
      console.error('Simple camera failed:', error);
      setCameraError(`Simple camera failed: ${error.message}`);
    }
  };

  // Handle file upload from input
  const handleFileUpload = (e) => {
    const fileInput = e.target;
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];
      console.log('File selected:', file.name, file.size, 'bytes');
      
      // Create both blob URL and data URL
      const imageUrl = URL.createObjectURL(file);
      
      // Convert file to data URL for localStorage
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        setCapturedImage({ 
          blob: file, 
          url: imageUrl,
          dataUrl: dataUrl 
        });
        setPhotoCaptured(true);
      };
      reader.readAsDataURL(file);
    }
    console.log('Photo uploaded via file input');
  };

  // Add debugging for capturedImage state
  useEffect(() => {
    console.log('capturedImage state changed:', capturedImage);
  }, [capturedImage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="user-report">
      {/* Header */}
      <header className="report-header">
        <h1>🚨 Report Billboard</h1>
        <p>Help keep our community safe by reporting unauthorized billboards</p>
      </header>

      {/* Success Message */}
      {showSuccess && (
        <div className="success-message">
          ✅ Report submitted successfully! Thank you for helping our community.
        </div>
      )}

      {/* Photo Captured Message */}
      {photoCaptured && (
        <div className="success-message">
          📸 Photo captured successfully! You can now fill out the report details below.
        </div>
      )}

      {/* Camera Error Message */}
      {cameraError && (
        <div className="error-message">
          ❌ {cameraError}
        </div>
      )}

      {/* Camera Section */}
      {isCameraOpen && (
        <div className="camera-container">
          <div className="camera-header">
            <h3>📸 Take Photo</h3>
            <button className="close-camera" onClick={closeCamera}>✕</button>
          </div>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted
            className="camera-video"
            style={{ width: '100%', height: 'auto', backgroundColor: '#000' }}
          />
          <div className="camera-controls">
            <button className="btn btn-primary" onClick={capturePhoto}>
              📸 Capture Photo
            </button>
            <button className="btn btn-secondary" onClick={openCamera}>
              🔄 Retry Camera
            </button>
            <button className="btn btn-secondary" onClick={closeCamera}>
              ❌ Cancel
            </button>
          </div>
        </div>
      )}

      {/* Photo Display */}
      {capturedImage && !isCameraOpen && (
        <div className="photo-section">
          <h3>📸 Captured Photo</h3>
          <div className="photo-preview">
            <img src={capturedImage.url} alt="Billboard detection" />
          </div>
          <div className="photo-actions">
            <button className="btn btn-secondary" onClick={retakePhoto}>
              📷 Retake Photo
            </button>
          </div>
        </div>
      )}

      {/* Report Form */}
      <div className="report-form">
        <h3>📝 Report Details</h3>
        
        {/* Photo Button */}
        {!capturedImage && (
          <div className="photo-button-section">
            <button 
              className="btn btn-primary photo-btn"
              onClick={handleTakePhoto}
            >
              📸 Take Photo of Billboard
            </button>
            
            {/* Hidden file input that opens camera */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            
            <div style={{ marginTop: '10px' }}>
              <button 
                className="btn btn-secondary" 
                onClick={() => openCamera()}
                style={{ fontSize: '0.8rem', marginRight: '10px' }}
              >
                🎥 Try Video Camera
              </button>
              
              <button 
                className="btn btn-secondary debug-btn" 
                onClick={debugCamera}
                style={{ fontSize: '0.8rem' }}
              >
                🔍 Debug
              </button>
            </div>
            
            <p className="photo-hint">📱 On mobile: Button opens camera directly</p>
            <p className="photo-hint">💻 On desktop: Choose file or try video camera</p>
          </div>
        )}

        {/* Description */}
        <div className="form-group">
          <label htmlFor="description">Description:</label>
          <textarea
            id="description"
            name="description"
            value={reportData.description}
            onChange={handleInputChange}
            placeholder="Describe what you see (e.g., 'Large unauthorized billboard blocking traffic view')"
            rows="3"
            required
          />
        </div>

        {/* Location */}
        <div className="form-group">
          <label htmlFor="location">Location:</label>
          <div className="location-input">
            <input
              type="text"
              id="location"
              name="location"
              value={reportData.location}
              onChange={handleInputChange}
              placeholder="Street address or coordinates"
              required
            />
            <button 
              type="button" 
              className="btn btn-secondary location-btn"
              onClick={getCurrentLocation}
            >
              📍 Use My Location
            </button>
          </div>
        </div>

        {/* Urgency */}
        <div className="form-group">
          <label htmlFor="urgency">Urgency Level:</label>
          <select
            id="urgency"
            name="urgency"
            value={reportData.urgency}
            onChange={handleInputChange}
          >
            <option value="low">Low - General violation</option>
            <option value="medium">Medium - Safety concern</option>
            <option value="high">High - Immediate hazard</option>
          </select>
        </div>

        {/* Consent */}
        <div className="form-group consent-group">
          <label className="consent-label">
            <input
              type="checkbox"
              name="consent"
              checked={reportData.consent}
              onChange={handleInputChange}
              required
            />
            <span className="checkmark"></span>
            I consent to submit this report and understand that my photo and location data will be used for civic enforcement purposes.
          </label>
        </div>

        {/* Submit Button */}
        <button
          className={`btn btn-primary submit-btn ${!capturedImage || !reportData.consent ? 'disabled' : ''}`}
          onClick={submitReport}
          disabled={!capturedImage || !reportData.consent || isSubmitting}
        >
          {isSubmitting ? '📤 Submitting...' : '📤 Submit Report'}
        </button>
      </div>

      {/* Instructions */}
      <div className="instructions">
        <h3>📋 How to Report:</h3>
        <ol>
          <li>📸 Take a clear photo of the unauthorized billboard</li>
          <li>📍 Provide the location (or use GPS)</li>
          <li>📝 Describe what you see</li>
          <li>🚨 Select urgency level</li>
          <li>✅ Give consent and submit</li>
        </ol>
        
        <div className="safety-tips">
          <h4>⚠️ Safety Tips:</h4>
          <ul>
            <li>Don't put yourself in danger to take a photo</li>
            <li>Stay on public property</li>
            <li>Don't confront anyone about the billboard</li>
            <li>Report from a safe distance</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
