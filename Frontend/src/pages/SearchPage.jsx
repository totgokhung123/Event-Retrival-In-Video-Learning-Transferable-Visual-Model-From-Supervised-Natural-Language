import React, { useState } from 'react';
import { Container, Typography, Box, CircularProgress, Alert, Grid, Button, Collapse } from '@mui/material';
import SearchForm from '../components/SearchForm';
import SearchHelp from '../components/SearchHelp';
import EventCard from '../components/EventCard';
import axios from 'axios';
import InfoIcon from '@mui/icons-material/Info';

const API_URL = 'http://localhost:5000/api';

const SearchPage = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  const handleSearch = async (searchParams) => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Search params:', searchParams);
      const response = await axios.post(`${API_URL}/search`, searchParams);
      setSearchResults(response.data.events || []);
      
      // Log the results for debugging
      console.log('Search results:', response.data);
      
      if (response.data.events && response.data.events.length === 0) {
        setError('No results found for your search criteria. Try adjusting your search parameters.');
      }
    } catch (error) {
      console.error('Search error:', error);
      setError(error.response?.data?.error || 'An error occurred during search. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Video Search Engine
      </Typography>
      
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          startIcon={<InfoIcon />}
          onClick={() => setShowHelp(!showHelp)}
          variant={showHelp ? "contained" : "outlined"}
          color="info"
          size="small"
        >
          {showHelp ? "Ẩn hướng dẫn" : "Xem hướng dẫn tìm kiếm"}
        </Button>
      </Box>
      
      <Collapse in={showHelp}>
        <SearchHelp />
      </Collapse>
      
      <SearchForm onSearch={handleSearch} isLoading={isLoading} />
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : searchResults.length > 0 ? (
        <>
          <Typography variant="h5" component="h2" gutterBottom>
            Search Results ({searchResults.length})
          </Typography>
          
          <Grid container spacing={3}>
            {searchResults.map((event) => (
              <Grid item xs={12} sm={6} md={4} key={event.id}>
                <EventCard event={event} />
              </Grid>
            ))}
          </Grid>
        </>
      ) : !error && (
        <Box sx={{ my: 4, textAlign: 'center' }}>
          <Typography variant="body1">
            Enter a search query or upload an image to find relevant video events.
          </Typography>
        </Box>
      )}
    </Container>
  );
};

export default SearchPage; 