import { useState, useEffect } from 'react'
import axios from 'axios'
import { TrainingMetrics } from './components/TrainingMetrics'
import {
  Box,
  Typography,
  Paper,
} from '@mui/material'
import './index.css'
import Layout from './components/Layout'
import { NetworkArchitectureEditor } from './components/NetworkArchitectureEditor'

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

    fetchMetrics()
  }, [])

  return (
    <Layout>
      <NetworkArchitectureEditor />
      <Paper
        elevation={3}
        sx={{
          p: 3,
          my: 2,
          backgroundColor: 'background.paper'
        }}
      >
        {error && (
          <Typography color="error" sx={{ mb: 2 }}>{error}</Typography>
        )}
        {metrics && (
          <TrainingMetrics data={metrics} />
        )}
        {!metrics && !error && (
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <Typography>Loading metrics...</Typography>
          </Box>
        )}
      </Paper>
    </Layout>
  )
}

export default App
