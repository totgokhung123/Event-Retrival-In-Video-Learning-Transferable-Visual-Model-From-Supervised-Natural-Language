import React from 'react';
import { Card, CardContent, CardMedia, Typography, Box, Chip, Button, Stack } from '@mui/material';
import { styled } from '@mui/material/styles';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CategoryIcon from '@mui/icons-material/Category';
import VerifiedIcon from '@mui/icons-material/Verified';

const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'scale(1.02)',
    boxShadow: theme.shadows[6],
  },
}));

const EventCard = ({ event }) => {
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Get confidence color based on value
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'primary';
    if (confidence >= 0.4) return 'warning';
    return 'error';
  };

  return (
    <StyledCard>
      <CardMedia
        component="img"
        height="160"
        image={`http://localhost:5000/api/frame/${event.thumbnailUrl}`}
        alt={event.title}
        sx={{ objectFit: 'cover' }}
        onError={(e) => {
          e.target.onerror = null;
          e.target.src = 'https://via.placeholder.com/320x160?text=No+Preview';
        }}
      />
      
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="h6" component="div" gutterBottom noWrap>
          {event.title}
        </Typography>
        
        <Typography variant="body2" color="text.secondary" gutterBottom sx={{ 
          height: '40px', 
          overflow: 'hidden', 
          textOverflow: 'ellipsis', 
          display: '-webkit-box', 
          WebkitLineClamp: 2, 
          WebkitBoxOrient: 'vertical' 
        }}>
          {event.description}
        </Typography>
        
        <Stack spacing={1} sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <VideoLibraryIcon fontSize="small" color="action" />
            <Typography variant="body2" color="text.secondary">
              {event.videoId.replace('video-', 'Video ')}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AccessTimeIcon fontSize="small" color="action" />
            <Typography variant="body2" color="text.secondary">
              {formatTime(event.timestamp)}
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CategoryIcon fontSize="small" color="action" />
            <Chip 
              label={event.category} 
              size="small" 
              variant="outlined" 
            />
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <VerifiedIcon fontSize="small" color="action" />
            <Chip 
              label={`Confidence: ${(event.confidence * 100).toFixed(0)}%`}
              size="small"
              color={getConfidenceColor(event.confidence)}
            />
          </Box>
        </Stack>
      </CardContent>
      
      <Box sx={{ p: 2, pt: 0 }}>
        <Button 
          variant="contained" 
          fullWidth
          onClick={() => window.open(`http://localhost:5000/api/video/${event.videoId.replace('video-', '')}#t=${event.timestamp}`)}
        >
          Play Video
        </Button>
      </Box>
    </StyledCard>
  );
};

export default EventCard; 