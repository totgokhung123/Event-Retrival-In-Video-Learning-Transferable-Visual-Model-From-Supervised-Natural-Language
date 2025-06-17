import React, { useState, useRef, useEffect } from 'react';
import { Search, Image, Mic, Upload, Sparkles, ChevronDown, ChevronUp, Settings, Zap, Eye, Volume2, StopCircle, Globe } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import axios from 'axios';
import { EventData } from '../types';

const API_URL = 'http://localhost:5000/api';

export const AdvancedSearchPanel = () => {
  const { currentVideo, setEvents, activeSearchFilters, updateActiveSearchFilters } = useVideo();
  const [searchType, setSearchType] = useState<'text' | 'image' | 'voice'>('text');
  const [textQuery, setTextQuery] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showFilters, setShowFilters] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  
  // Voice recording states
  const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState<'vi' | 'en_us'>('en_us');
  
  // Refs for audio recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // Detailed filter states - sử dụng filters từ context
  const [filters, setFilters] = useState({
    enableTextKeyword: activeSearchFilters.enableTextKeyword || false,
    textKeyword: activeSearchFilters.textKeyword || '',
    minTextConfidence: activeSearchFilters.minTextConfidence || 0.0,
    enableObjectKeyword: activeSearchFilters.enableObjectKeyword || false,
    objectKeyword: activeSearchFilters.objectKeyword || '',
    minObjectConfidence: activeSearchFilters.minObjectConfidence || 0.0,
    enableClipSimilarity: activeSearchFilters.enableClipSimilarity || false,
    minSimilarity: activeSearchFilters.minSimilarity || 0.0,
  });

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle audio file upload
  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Check if file is too large (approximating 30 seconds of audio)
      if (file.size > 1024 * 1024 * 2) { // 2MB limit as approximation
        setSearchError('Audio file too large. Please use a recording less than 30 seconds.');
        return;
      }
      setRecordedAudio(file);
      setSearchError(null);
    }
  };

  // Start voice recording
  const startVoiceRecording = async () => {
    setSearchError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/mp3' });
        setRecordedAudio(audioBlob);
        
        // Stop all tracks from the stream
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Start recording
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start a timer that counts up to 30 seconds
      timerRef.current = setInterval(() => {
        setRecordingTime(prevTime => {
          if (prevTime >= 30) {
            stopVoiceRecording();
            return 30;
          }
          return prevTime + 1;
        });
      }, 1000);
      
      // Auto-stop after 30 seconds
      setTimeout(() => {
        if (isRecording && mediaRecorderRef.current?.state === 'recording') {
          stopVoiceRecording();
        }
      }, 30000);
      
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setSearchError('Could not access microphone. Please check permissions.');
    }
  };
  
  // Stop voice recording
  const stopVoiceRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    // Clear the timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    setIsRecording(false);
  };
  
  // Cleanup function for recording
  useEffect(() => {
    return () => {
      // Clean up when component unmounts
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);
  
  // Transcribe audio using API
  const transcribeAudio = async () => {
    if (!recordedAudio) return;
    
    setIsTranscribing(true);
    setSearchError(null);
    
    try {
      // Create a FormData object to send the audio file
      const formData = new FormData();
      formData.append('audio', recordedAudio);
      formData.append('language', selectedLanguage);
      
      // Send to backend for transcription
      const response = await axios.post(`${API_URL}/transcribe-voice`, formData);
      
      // Update the search query with transcribed text
      const transcribedText = response.data.text || '';
      setTranscribedText(transcribedText);
      setTextQuery(transcribedText);
      
    } catch (err) {
      console.error('Transcription error:', err);
      if (axios.isAxiosError(err) && err.response) {
        setSearchError(`Transcription failed: ${err.response.data.error || 'Unknown error'}`);
      } else {
        setSearchError('Transcription failed. Please try again.');
      }
    } finally {
      setIsTranscribing(false);
    }
  };
  
  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

  const updateFilters = (key: string, value: unknown): void => {
    setFilters((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  // Determine search method based on filters and search query
  const determineSearchMethod = (): string => {
    const { enableTextKeyword, enableObjectKeyword, enableClipSimilarity } = filters;
    const hasSearchQuery = textQuery.trim().length > 0;
    
    // Case 7: Search Query + Object-Keyword + Text-Keyword
    if (hasSearchQuery && enableTextKeyword && enableObjectKeyword) {
      return 'text_object_keyword';
    }
    
    // Case 6: Search Query + Object-Keyword
    if (hasSearchQuery && enableObjectKeyword && !enableTextKeyword) {
      return 'text_object';
    }
    
    // Case 5: Only Object-Keyword
    if (!hasSearchQuery && enableObjectKeyword && !enableTextKeyword) {
      return 'object_only';
    }
    
    // Case 4: Search Query + Text-Keyword
    if (hasSearchQuery && enableTextKeyword && !enableObjectKeyword) {
      return 'text_keyword';
    }
    
    // Case 3: Only Text-Keyword
    if (!hasSearchQuery && enableTextKeyword && !enableObjectKeyword) {
      return 'keyword_only';
    }
    
    // Case 2: Search with CLIP Similarity Threshold
    if (hasSearchQuery && enableClipSimilarity) {
      return 'text_adaptive';
    }
    
    // Case 1: Default - just text
    if (hasSearchQuery) {
      return 'text_clip';
    }
    
    // Fallback
    return 'text_clip';
  };

  // Modify handleSearch to incorporate voice search
  const handleSearch = async () => {
    // If we're in voice mode and have recorded audio but haven't transcribed it yet
    if (searchType === 'voice' && recordedAudio && !transcribedText) {
      await transcribeAudio();
      return; // Don't continue with search until transcription is complete
    }
    
    // If we're in voice mode and have transcribed text, use it for search
    if (searchType === 'voice' && transcribedText) {
      // Switch to text search mode but keep the transcribed text
      setSearchType('text');
    }
    
    // Update the global context with current filter settings
    updateActiveSearchFilters(filters);
    
    // Cho phép tìm kiếm khi có ít nhất một trong các trường: textQuery, textKeyword hoặc objectKeyword
    if (searchType === 'text' && 
        !textQuery && 
        !(filters.enableTextKeyword && filters.textKeyword) && 
        !(filters.enableObjectKeyword && filters.objectKeyword)) {
      setSearchError('Please enter a search query or enable a keyword filter');
      return;
    }

    if (!imageUrl && searchType === 'image') {
      setSearchError('Please provide an image URL or upload an image');
      return;
    }

    // Đặt state loading trước khi bắt đầu tìm kiếm
    setIsSearching(true);
    setSearchError(null);
    
    // Thêm một delay nhỏ để đảm bảo UI hiển thị trạng thái loading
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
      // Build search parameters based on search type and filters
      const searchParams: Record<string, unknown> = {
        top_k: 20, // Default number of results
      };
      
      // Set specific confidence thresholds based on search method
      const searchMethod = determineSearchMethod();
      console.log("Selected search method:", searchMethod);
      
      // Luôn thiết lập cả 3 threshold để đảm bảo backend trả về đầy đủ thông tin confidence
      searchParams.adaptive_threshold = filters.minSimilarity; // CLIP similarity threshold
      searchParams.text_confidence = filters.minTextConfidence; // Text detection threshold
      searchParams.object_confidence = filters.minObjectConfidence; // Object detection threshold
      
      // Thêm tham số để yêu cầu backend trả về đầy đủ thông tin confidence
      searchParams.return_all_confidences = true;
      
      // Set search type and query based on input
      if (searchType === 'text') {
        searchParams.search_type = 'text';
        const searchMethod = determineSearchMethod();
        searchParams.search_method = searchMethod;
        
        // Set appropriate query text based on search method
        if (searchMethod === 'keyword_only' && filters.enableTextKeyword) {
          // Case 3: Only Text-Keyword
          searchParams.query = filters.textKeyword || '';
        } else if (searchMethod === 'object_only' && filters.enableObjectKeyword) {
          // Case 5: Only Object-Keyword
          searchParams.query = filters.objectKeyword || '';
        } else if (searchMethod === 'text_object' && filters.enableObjectKeyword) {
          // Case 6: Text + Object-Keyword
          searchParams.query = textQuery;
          searchParams.object = filters.objectKeyword || '';
          console.log("Using object keyword:", filters.objectKeyword);
        } else if (searchMethod === 'text_keyword' && filters.enableTextKeyword) {
          // Case 4: Text + Text-Keyword
          searchParams.query = textQuery;
          searchParams.keyword = filters.textKeyword || '';
          console.log("Using text keyword:", filters.textKeyword);
        } else if (searchMethod === 'text_object_keyword') {
          // Case 7: Combined - prioritize main search query
          searchParams.query = textQuery;
          searchParams.keyword = filters.textKeyword || '';
          searchParams.object = filters.objectKeyword || '';
          console.log("Using combined keywords - text:", filters.textKeyword, "object:", filters.objectKeyword);
        } else {
          // Other cases: use main search query
          searchParams.query = textQuery;
        }

      } else if (searchType === 'image') {
        searchParams.search_type = 'image';
        searchParams.image_url = imageUrl;
      }

      // Call the API
      console.log('Searching with params:', searchParams);
      const response = await axios.post(`${API_URL}/search`, searchParams);
      
      // Type assertion for response data
      interface SearchResponse {
        events: EventData[];
      }
      
      // Update the events in the video context to display the results
      const responseData = response.data as SearchResponse;
      if (responseData.events && responseData.events.length > 0) {
        setEvents(responseData.events);
      } else {
        setSearchError('No results found. Try adjusting your search parameters.');
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchError('An error occurred during search. Please try again.');
    } finally {
      // Đảm bảo luôn reset state loading khi hoàn thành
      setIsSearching(false);
    }
  };

  // Enhance handleClearSearch to properly reset all voice-related states
  const handleClearSearch = () => {
    setTextQuery('');
    setImageUrl('');
    setIsRecording(false);
    setRecordedAudio(null);
    setTranscribedText('');
    setRecordingTime(0);
    // Reset events in the video context
    setEvents([]);
    setSearchError(null);
  };

  // Custom checkbox component
  const CustomCheckbox = ({ checked, onChange, label, icon, color = 'teal' }: {
    id?: string; // Optional id
    checked: boolean;
    onChange: (checked: boolean) => void;
    label: string;
    icon: React.ReactNode;
    color?: string;
  }) => {
    const colorClasses = {
      teal: 'from-teal-500 to-cyan-500',
      purple: 'from-purple-500 to-pink-500',
      orange: 'from-orange-500 to-red-500'
    };

    return (
      <div className="flex items-center gap-3">
        <div className="relative">
          <button
            type="button"
            onClick={() => onChange(!checked)}
            className={`w-5 h-5 rounded border-2 transition-all duration-300 flex items-center justify-center ${
              checked
                ? `bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} border-transparent shadow-lg`
                : 'bg-slate-700 border-slate-600 hover:border-slate-500'
            }`}
          >
            {checked && (
              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
          </button>
          {checked && (
            <div className={`absolute inset-0 rounded bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} opacity-20 animate-pulse`}></div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {icon}
          <label className="text-sm font-medium text-slate-300 cursor-pointer" onClick={() => onChange(!checked)}>
            {label}
          </label>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-slate-800 to-slate-900">
      {/* Advanced Search Section */}
      <div className="p-4 border-b border-slate-700/50 backdrop-blur-sm">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center shadow-lg">
            <Sparkles size={16} className="text-white" />
          </div>
          <h2 className="text-lg font-semibold bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent">
            Advanced Search
          </h2>
        </div>

        {/* Search Type Tabs */}
        <div className="flex gap-1 mb-4 bg-slate-700/50 backdrop-blur-sm rounded-xl p-1 border border-slate-600/30">
          <button
            onClick={() => setSearchType('text')}
            className={`flex-1 flex items-center justify-center gap-2 py-3 px-3 rounded-lg text-sm font-medium transition-all duration-300 ${
              searchType === 'text'
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg transform scale-105'
                : 'text-slate-400 hover:text-white hover:bg-slate-600/50'
            }`}
          >
            <Search size={14} />
            Text
          </button>
          <button
            onClick={() => setSearchType('image')}
            className={`flex-1 flex items-center justify-center gap-2 py-3 px-3 rounded-lg text-sm font-medium transition-all duration-300 ${
              searchType === 'image'
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg transform scale-105'
                : 'text-slate-400 hover:text-white hover:bg-slate-600/50'
            }`}
          >
            <Image size={14} />
            Image
          </button>
          <button
            onClick={() => setSearchType('voice')}
            className={`flex-1 flex items-center justify-center gap-2 py-3 px-3 rounded-lg text-sm font-medium transition-all duration-300 ${
              searchType === 'voice'
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 text-white shadow-lg transform scale-105'
                : 'text-slate-400 hover:text-white hover:bg-slate-600/50'
            }`}
          >
            <Mic size={14} />
            Voice
          </button>
        </div>

        {/* Search Content */}
        <div className="space-y-4">
          {searchType === 'text' && (
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-300">
                Search Query
              </label>
              <div className="relative group">
                <input
                  type="text"
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  placeholder="Describe what you're looking for..."
                  className="w-full p-4 bg-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl text-sm focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all duration-300 group-hover:border-slate-500"
                />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-teal-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
              </div>
              <p className="text-xs text-slate-400 flex items-center gap-2">
                <Zap size={12} className="text-teal-400" />
                e.g., "person crossing street", "red car stopping"
              </p>
            </div>
          )}

          {searchType === 'image' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Image URL
                </label>
                <div className="relative group">
                  <input
                    type="url"
                    value={imageUrl}
                    onChange={(e) => setImageUrl(e.target.value)}
                    placeholder="https://example.com/image.jpg"
                    className="w-full p-4 bg-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-xl text-sm focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all duration-300"
                  />
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-teal-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                </div>
              </div>
              
              <div className="text-center">
                <div className="inline-flex items-center gap-2 text-slate-400 text-sm">
                  <div className="h-px bg-gradient-to-r from-transparent via-slate-600 to-transparent flex-1"></div>
                  <span>or</span>
                  <div className="h-px bg-gradient-to-r from-transparent via-slate-600 to-transparent flex-1"></div>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Upload Image
                </label>
                <div className="relative group">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                    id="image-upload"
                  />
                  <label
                    htmlFor="image-upload"
                    className="flex items-center justify-center gap-3 w-full p-6 border-2 border-dashed border-slate-600/50 rounded-xl cursor-pointer hover:border-teal-500/50 transition-all duration-300 bg-slate-700/20 hover:bg-slate-700/40 group"
                  >
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center group-hover:from-teal-500/30 group-hover:to-cyan-500/30 transition-all duration-300">
                      <Upload size={16} className="text-teal-400" />
                    </div>
                    <span className="text-slate-300 font-medium">Choose image file</span>
                  </label>
                </div>
              </div>
              
              {imageUrl && (
                <div className="mt-4 relative group">
                  <img
                    src={imageUrl}
                    alt="Search reference"
                    className="w-full h-32 object-cover rounded-xl border border-slate-600/50"
                  />
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </div>
              )}
            </div>
          )}

          {searchType === 'voice' && (
            <div className="space-y-6 py-2">
              {/* Language selector */}
              <div className="flex items-center space-x-3 mb-2">
                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
                  <Globe size={12} className="text-purple-400" />
                </div>
                <label className="block text-sm font-medium text-slate-300">
                  Language
                </label>
                <select 
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value as 'vi' | 'en_us')}
                  className="ml-2 p-2 text-sm bg-slate-700 text-slate-200 rounded-lg border border-slate-600 focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500"
                >
                  <option value="vi">VN</option>
                  <option value="en_us">ENG</option>
                </select>
              </div>
              
              {/* Upload audio file section */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Upload Voice Recording
                </label>
                <div className="relative group">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioUpload}
                    className="hidden"
                    id="audio-upload"
                    disabled={isRecording || isTranscribing}
                  />
                  <label
                    htmlFor="audio-upload"
                    className={`flex items-center justify-center gap-3 w-full p-6 border-2 border-dashed border-slate-600/50 rounded-xl cursor-pointer hover:border-teal-500/50 transition-all duration-300 bg-slate-700/20 hover:bg-slate-700/40 ${(isRecording || isTranscribing) ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center group-hover:from-teal-500/30 group-hover:to-cyan-500/30 transition-all duration-300">
                      <Volume2 size={16} className="text-teal-400" />
                    </div>
                    <span className="text-slate-300 font-medium">Upload audio file</span>
                  </label>
                </div>
                <p className="text-xs text-slate-400 mt-2 flex items-center gap-2">
                  <Zap size={12} className="text-teal-400" />
                  Max duration: 30 seconds
                </p>
              </div>

              <div className="text-center">
                <div className="inline-flex items-center gap-2 text-slate-400 text-sm">
                  <div className="h-px bg-gradient-to-r from-transparent via-slate-600 to-transparent flex-1"></div>
                  <span>or</span>
                  <div className="h-px bg-gradient-to-r from-transparent via-slate-600 to-transparent flex-1"></div>
                </div>
              </div>
              
              {/* Record voice section */}
              <div className="text-center space-y-6 py-2">
                <div className="flex flex-col items-center">
                  <div className="relative">
                    {!isRecording ? (
                      <button
                        onClick={startVoiceRecording}
                        disabled={isRecording || isTranscribing || !!recordedAudio}
                        className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-2xl ${
                          isTranscribing || !!recordedAudio
                            ? 'bg-slate-600 cursor-not-allowed'
                            : 'bg-gradient-to-br from-teal-500 to-cyan-600 hover:from-teal-400 hover:to-cyan-500 hover:scale-110'
                        }`}
                      >
                        <Mic size={28} className="text-white" />
                      </button>
                    ) : (
                      <button
                        onClick={stopVoiceRecording}
                        className="w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-2xl bg-gradient-to-br from-red-500 to-pink-600 hover:from-red-400 hover:to-pink-500 hover:scale-110"
                      >
                        <StopCircle size={28} className="text-white" />
                      </button>
                    )}
                    {isRecording && (
                      <div className="absolute inset-0 rounded-full border-4 border-red-400/30 animate-ping"></div>
                    )}
                  </div>
                  
                  <p className="text-sm text-slate-300 mt-4 font-medium flex items-center justify-center gap-2">
                    {isRecording ? (
                      <>
                        <span className="text-red-400">Recording</span> 
                        <span className="bg-red-500/20 px-2 py-1 rounded font-mono">{formatTime(recordingTime)}/00:30</span>
                      </>
                    ) : recordedAudio ? (
                      <span className="text-green-400">Audio Ready</span>
                    ) : (
                      <span>Click to start recording</span>
                    )}
                  </p>
                </div>
                
                {isRecording && (
                  <div className="flex justify-center">
                    <div className="flex space-x-2">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className="w-1 bg-gradient-to-t from-teal-500 to-cyan-400 rounded-full animate-pulse"
                          style={{
                            height: `${Math.random() * 32 + 16}px`,
                            animationDelay: `${i * 0.1}s`
                          }}
                        ></div>
                      ))}
                    </div>
                  </div>
                )}
                
                {recordedAudio && (
                  <div className="mt-4 flex flex-col items-center gap-4">
                    <audio 
                      controls 
                      src={recordedAudio instanceof Blob ? URL.createObjectURL(recordedAudio) : undefined}
                      className="w-full max-w-xs"
                    />
                    <button
                      onClick={() => {
                        setRecordedAudio(null);
                        setTranscribedText('');
                        setSearchError(null);
                      }}
                      className="text-xs text-slate-400 hover:text-slate-300"
                    >
                      Clear recording
                    </button>
                  </div>
                )}
              </div>
              
              {transcribedText && (
                <div className="mt-4 p-4 bg-slate-700/50 rounded-lg border border-teal-500/30">
                  <p className="text-sm text-slate-200">
                    <span className="font-medium text-teal-400">Transcribed Text:</span> {transcribedText}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
              <Settings size={12} className="text-purple-400" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">Advanced Filters</h3>
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="text-slate-400 hover:text-white transition-all duration-300 hover:scale-110"
          >
            {showFilters ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {showFilters && (
          <div className="space-y-4 border border-slate-600/30 rounded-xl p-4 bg-gradient-to-br from-slate-700/20 to-slate-800/20 backdrop-blur-sm">
            {/* Text-Keyword Matching */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-700/20 border border-slate-600/20 hover:border-slate-500/30 transition-all duration-300">
              <CustomCheckbox
                id="enableTextKeyword"
                checked={filters.enableTextKeyword}
                onChange={(checked) => updateFilters('enableTextKeyword', checked)}
                label="Text-Keyword Matching"
                icon={<Eye size={14} className="text-teal-400" />}
                color="teal"
              />
              
              {filters.enableTextKeyword && (
                <div className="ml-8 space-y-4 animate-in slide-in-from-top-2 duration-300">
                  <div>
                    <label className="block text-xs text-slate-400 mb-2 font-medium">Keyword:</label>
                    <div className="relative group">
                      <input
                        type="text"
                        value={filters.textKeyword}
                        onChange={(e) => updateFilters('textKeyword', e.target.value)}
                        className="w-full p-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-xs focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all duration-300"
                        placeholder="Enter keyword..."
                      />
                      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-teal-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-2 font-medium">
                      Min Text-Detection Confidence: 
                      <span className="text-teal-400 font-semibold ml-1">
                        {Math.round(filters.minTextConfidence * 100)}%
                      </span>
                    </label>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={filters.minTextConfidence}
                        onChange={(e) => updateFilters('minTextConfidence', parseFloat(e.target.value))}
                        className="w-full h-2 bg-slate-600/50 rounded-lg appearance-none cursor-pointer slider-teal"
                      />
                      <div 
                        className="absolute top-0 h-2 bg-gradient-to-r from-teal-500 to-cyan-500 rounded-lg pointer-events-none"
                        style={{ width: `${filters.minTextConfidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Object-Keyword Matching */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-700/20 border border-slate-600/20 hover:border-slate-500/30 transition-all duration-300">
              <CustomCheckbox
                id="enableObjectKeyword"
                checked={filters.enableObjectKeyword}
                onChange={(checked) => updateFilters('enableObjectKeyword', checked)}
                label="Object-Keyword Matching"
                icon={<Search size={14} className="text-purple-400" />}
                color="purple"
              />
              
              {filters.enableObjectKeyword && (
                <div className="ml-8 space-y-4 animate-in slide-in-from-top-2 duration-300">
                  <div>
                    <label className="block text-xs text-slate-400 mb-2 font-medium">Keyword:</label>
                    <div className="relative group">
                      <input
                        type="text"
                        value={filters.objectKeyword}
                        onChange={(e) => updateFilters('objectKeyword', e.target.value)}
                        className="w-full p-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-xs focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 transition-all duration-300"
                        placeholder="Enter object keyword..."
                      />
                      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-500/5 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-2 font-medium">
                      Min Object-Detection Confidence: 
                      <span className="text-purple-400 font-semibold ml-1">
                        {Math.round(filters.minObjectConfidence * 100)}%
                      </span>
                    </label>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={filters.minObjectConfidence}
                        onChange={(e) => updateFilters('minObjectConfidence', parseFloat(e.target.value))}
                        className="w-full h-2 bg-slate-600/50 rounded-lg appearance-none cursor-pointer"
                      />
                      <div 
                        className="absolute top-0 h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg pointer-events-none"
                        style={{ width: `${filters.minObjectConfidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* CLIP Similarity Threshold */}
            <div className="space-y-3 p-4 rounded-xl bg-slate-700/20 border border-slate-600/20 hover:border-slate-500/30 transition-all duration-300">
              <CustomCheckbox
                id="enableClipSimilarity"
                checked={filters.enableClipSimilarity}
                onChange={(checked) => updateFilters('enableClipSimilarity', checked)}
                label="CLIP Similarity Threshold"
                icon={<Zap size={14} className="text-orange-400" />}
                color="orange"
              />
              
              {filters.enableClipSimilarity && (
                <div className="ml-8 space-y-4 animate-in slide-in-from-top-2 duration-300">
                  <div>
                    <label className="block text-xs text-slate-400 mb-2 font-medium">
                      Min Similarity: 
                      <span className="text-orange-400 font-semibold ml-1">
                        {Math.round(filters.minSimilarity * 100)}%
                      </span>
                    </label>
                    <div className="relative">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={filters.minSimilarity}
                        onChange={(e) => updateFilters('minSimilarity', parseFloat(e.target.value))}
                        className="w-full h-2 bg-slate-600/50 rounded-lg appearance-none cursor-pointer"
                      />
                      <div 
                        className="absolute top-0 h-2 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg pointer-events-none"
                        style={{ width: `${filters.minSimilarity * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      
        {/* Display error message if there is one */}
        {searchError && (
          <div className="mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg">
            <p className="text-sm text-red-300">{searchError}</p>
          </div>
        )}
      
        <div className="pt-6 flex justify-end gap-3">
          <button
            onClick={handleClearSearch}
            className="px-4 py-2 border border-slate-600/50 rounded-lg text-sm hover:bg-slate-700/50 transition-all duration-300 text-slate-300"
            disabled={isSearching || isTranscribing}
          >
            Clear
          </button>
          <button 
            onClick={handleSearch}
            disabled={isSearching || isTranscribing || (searchType === 'voice' && !recordedAudio)}
            className={`px-6 py-2 rounded-lg text-sm text-white transition-all duration-300 shadow-lg font-medium ${
              isSearching || isTranscribing
                ? 'bg-slate-600 cursor-not-allowed' 
                : searchType === 'voice' && recordedAudio && !transcribedText
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-400 hover:to-pink-400 hover:shadow-purple-500/25'
                  : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 hover:shadow-teal-500/25'
            }`}
          >
            {isTranscribing 
              ? 'Transcribing...' 
              : isSearching 
                ? 'Searching...' 
                : searchType === 'voice' && recordedAudio && !transcribedText
                  ? 'Transcribe'
                  : 'Search'
            }
          </button>
        </div>
      </div>

      {/* Results Summary */}
      <div className="flex-1 p-4">
        {!currentVideo ? (
          <div className="flex items-center justify-center h-full text-center">
            <div className="space-y-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center mx-auto">
                <Eye size={24} className="text-slate-500" />
              </div>
              <p className="text-slate-500">
                Select a video to see detected events
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-teal-500/20 to-cyan-500/20 flex items-center justify-center mx-auto">
              <Settings size={20} className="text-teal-400" />
            </div>
            <p className="text-sm text-slate-400">
              Configure your search parameters above.
              <br />
              <span className="text-teal-400">Results will appear in the main timeline.</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};