import { useEffect, useRef } from 'react'
import { Box } from '@mui/material'

interface NetworkArchitecture {
  [key: string]: {
    n_inputs: number
    n_neurons: number
    activation: string | null
  }
}

interface NetworkDiagramProps {
  architecture: NetworkArchitecture
}

export const NetworkDiagram = ({ architecture }: NetworkDiagramProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set drawing styles
    ctx.strokeStyle = '#646cff'
    ctx.fillStyle = '#535bf2'
    ctx.lineWidth = 2

    const layers = Object.entries(architecture)
    const layerSpacing = canvas.width / (layers.length + 1)
    const nodeRadius = 10
    const maxNeurons = Math.max(...layers.map(([_, config]) => config.n_neurons))
    const verticalSpacing = canvas.height / (maxNeurons + 1)

    // Draw layers and connections
    layers.forEach((layer, layerIndex) => {
      const [_, config] = layer
      const x = layerSpacing * (layerIndex + 1)
      
      // Draw nodes for this layer
      for (let i = 0; i < config.n_neurons; i++) {
        const y = verticalSpacing * (i + 1)
        
        // Draw node
        ctx.beginPath()
        ctx.arc(x, y, nodeRadius, 0, Math.PI * 2)
        ctx.fill()

        // Draw connections to previous layer
        if (layerIndex > 0) {
          const prevLayer = layers[layerIndex - 1][1]
          const prevX = layerSpacing * layerIndex
          
          for (let j = 0; j < prevLayer.n_neurons; j++) {
            const prevY = verticalSpacing * (j + 1)
            ctx.beginPath()
            ctx.moveTo(prevX + nodeRadius, prevY)
            ctx.lineTo(x - nodeRadius, y)
            ctx.stroke()
          }
        }
      }
    })
  }, [architecture])

  return (
    <Box sx={{ mt: 3, textAlign: 'center' }}>
      <canvas 
        ref={canvasRef}
        width={600}
        height={400}
        style={{
          maxWidth: '100%',
          backgroundColor: '#1a1a1a',
          borderRadius: '8px'
        }}
      />
    </Box>
  )
} 