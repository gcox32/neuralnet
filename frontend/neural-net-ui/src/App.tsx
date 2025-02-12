import { useState, useEffect } from 'react'
import axios from 'axios'
import { TrainingMetrics } from './components/TrainingMetrics'
import { Box, Container, Typography, Paper } from '@mui/material'

interface TrainingData {
  iterations: number[]
  loss: number[]
  accuracy: number[]
  learning_rate: number[]
}

function App() {
  const [metrics, setMetrics] = useState<TrainingData | null>(null)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        console.log('Fetching metrics...')
        const response = await axios.get('http://127.0.0.1:5000/api/metrics', {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          withCredentials: false
        })
        console.log('Response:', response.data)
        setMetrics(response.data)
      } catch (err: any) {
        console.error('Full error:', err)
        console.error('Error response:', err.response)
        console.error('Error message:', err.message)
        setError(`Failed to fetch training metrics: ${err.message}`)
      }
    }

    // Test the server connection
    axios.get('http://127.0.0.1:5000/api/test', {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      withCredentials: false
    })
      .then(response => console.log('Server test:', response.data))
      .catch(err => console.error('Server test failed:', err))

    fetchMetrics()
  }, [])

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Neural Network Training Dashboard
        </Typography>
        
        <Paper elevation={3} sx={{ p: 3, my: 2 }}>
          {error && (
            <Typography color="error">{error}</Typography>
          )}
          {metrics && (
            <TrainingMetrics data={metrics} />
          )}
          {!metrics && !error && (
            <Typography>Loading metrics...</Typography>
          )}
        </Paper>
      </Box>
    </Container>
  )
}

export default App
