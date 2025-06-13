export interface VideoData {
  id: string;
  title: string;
  thumbnail: string;
  duration: number;
  uploadDate: string;
  size: string;
  resolution: string;
}

export interface EventData {
  id: string;
  videoId: string;
  title: string;
  description: string;
  timestamp: number;
  duration: number;
  category: string;
  confidence: number;
  thumbnailUrl: string;
}

export interface SearchFilter {
  id: string;
  name: string;
  type: 'select' | 'range' | 'checkbox' | 'radio';
  options?: string[];
  min?: number;
  max?: number;
}