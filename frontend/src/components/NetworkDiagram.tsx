import { useEffect, useRef, useState } from 'react'
import { Box, Switch, FormControlLabel, useTheme } from '@mui/material'
import { ACTIVATION_COLORS, NetworkArchitecture } from '../types/network'

interface NetworkDiagramProps {
    architecture: NetworkArchitecture
}

export const NetworkDiagram = ({ architecture }: NetworkDiagramProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [showActivations, setShowActivations] = useState(true)
    const [showLabels, setShowLabels] = useState(true)
    const [showXY, setShowXY] = useState(true)
    const theme = useTheme()

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        const layers = Object.entries(architecture)
        const layerSpacing = canvas.width / (layers.length + 1)
        const nodeRadius = 10
        const maxNeurons = Math.max(...layers.map(([_, config]) => config.n_neurons))
        const verticalSpacing = canvas.height / (maxNeurons + 1)

        // Draw layers and connections
        layers.forEach((layer, layerIndex) => {
            const [layerName, config] = layer
            const x = layerSpacing * (layerIndex + 1)

            // Draw activation function label if enabled
            if (showActivations && layerIndex > 0) {
                ctx.font = '12px Arial'
                ctx.fillStyle = theme.palette.text.primary
                ctx.textAlign = 'center'
                const activation = config.activation || 'linear'
                ctx.fillText(activation, x, 25)
            }

            // Draw nodes for this layer
            const layerHeight = (config.n_neurons - 1) * verticalSpacing;
            const startY = (canvas.height - layerHeight) / 2;

            for (let i = 0; i < config.n_neurons; i++) {
                const y = startY + (i * verticalSpacing);

                // Draw node with activation-specific color
                ctx.beginPath()
                ctx.arc(x, y, nodeRadius, 0, Math.PI * 2)
                if (showActivations) {
                    ctx.fillStyle = ACTIVATION_COLORS[config.activation || 'null']
                } else {
                    ctx.fillStyle = '#535bf2'
                }
                ctx.fill()

                // Draw input labels and arrows for input layer
                if (layerIndex === 0 && showXY) {
                    ctx.font = '14px Arial'
                    ctx.fillStyle = theme.palette.text.primary
                    ctx.textAlign = 'right'

                    // Using Unicode subscript numbers (₁, ₂, ₃, etc.) and italic x
                    const subscripts = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
                    ctx.font = 'italic 14px Arial'
                    ctx.fillText('x', x - nodeRadius * 4.2, y)
                    ctx.font = '14px Arial'
                    ctx.fillText(subscripts[i], x - nodeRadius * 3.8, y)

                    // Draw arrow with more margin
                    ctx.beginPath()
                    ctx.moveTo(x - nodeRadius * 3.2, y)
                    ctx.lineTo(x - nodeRadius * 1.7, y)
                    ctx.strokeStyle = theme.palette.text.primary
                    ctx.lineWidth = 1
                    ctx.stroke()

                    // Draw arrowhead
                    ctx.beginPath()
                    ctx.moveTo(x - nodeRadius * 1.7 - 5, y - 5)
                    ctx.lineTo(x - nodeRadius * 1.7, y)
                    ctx.lineTo(x - nodeRadius * 1.7 - 5, y + 5)
                    ctx.stroke()
                }

                // Draw output labels and arrows for output layer
                if (layerIndex === layers.length - 1 && showXY) {
                    ctx.font = '14px Arial'
                    ctx.fillStyle = theme.palette.text.primary
                    ctx.textAlign = 'left'

                    // Using Unicode subscript numbers and italic y
                    const subscripts = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
                    ctx.font = 'italic 14px Arial'
                    ctx.fillText('y', x + nodeRadius * 3.8, y)
                    ctx.font = '14px Arial'
                    ctx.fillText(subscripts[i], x + nodeRadius * 4.2, y)

                    // Draw arrow with more margin
                    ctx.beginPath()
                    ctx.moveTo(x + nodeRadius * 1.7, y)
                    ctx.lineTo(x + nodeRadius * 3.2, y)
                    ctx.strokeStyle = theme.palette.text.primary
                    ctx.lineWidth = 1
                    ctx.stroke()

                    // Draw arrowhead
                    ctx.beginPath()
                    ctx.moveTo(x + nodeRadius * 3.2 - 5, y - 5)
                    ctx.lineTo(x + nodeRadius * 3.2, y)
                    ctx.lineTo(x + nodeRadius * 3.2 - 5, y + 5)
                    ctx.stroke()
                }

                // Draw layer name below bottom node if it's the last node
                if (i === config.n_neurons - 1 && showLabels) {
                    ctx.font = '12px Arial'
                    ctx.fillStyle = theme.palette.text.primary
                    ctx.textAlign = 'center'
                    ctx.fillText(layerName, x, y + 25)
                }

                // Draw connections to previous layer
                if (layerIndex > 0) {
                    const prevLayer = layers[layerIndex - 1][1]
                    const prevX = layerSpacing * layerIndex
                    const prevLayerHeight = (prevLayer.n_neurons - 1) * verticalSpacing;
                    const prevStartY = (canvas.height - prevLayerHeight) / 2;

                    for (let j = 0; j < prevLayer.n_neurons; j++) {
                        const prevY = prevStartY + (j * verticalSpacing)
                        ctx.beginPath()
                        ctx.moveTo(prevX + nodeRadius, prevY)
                        ctx.lineTo(x - nodeRadius, y)
                        ctx.strokeStyle = '#646cff'
                        ctx.lineWidth = 1
                        ctx.stroke()
                    }
                }
            }
        })
    }, [architecture, showActivations, showLabels, showXY, theme.palette.text.primary])

    return (
        <Box sx={{ mt: 3, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

            <canvas
                ref={canvasRef}
                width={800}
                height={400}
                style={{
                    maxWidth: '100%',
                    backgroundColor: theme.palette.background.paper,
                    borderRadius: '8px'
                }}
            />
            <Box sx={{ display: 'flex', flexDirection: 'row', justifyContent: 'center' }}>
                <FormControlLabel
                    control={
                        <Switch
                            checked={showLabels}
                            onChange={(e) => setShowLabels(e.target.checked)}
                        />
                    }
                    label="Layer Labels"
                    sx={{ mt: 2 }}
                />
                <FormControlLabel
                    control={
                        <Switch
                            checked={showActivations}
                            onChange={(e) => setShowActivations(e.target.checked)}
                        />
                    }
                    label="Activation Functions"
                    sx={{ mt: 2 }}
                />
                <FormControlLabel
                    control={
                        <Switch
                            checked={showXY}
                            onChange={(e) => setShowXY(e.target.checked)}
                        />
                    }
                    label="Xs and Ys"
                    sx={{ mt: 2 }}
                />
            </Box>
        </Box>
    )
} 