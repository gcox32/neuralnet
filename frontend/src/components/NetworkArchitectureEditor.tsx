import { useState, useEffect } from 'react'
import {
    Box,
    Paper,
    Typography,
    Button,
    TextField,
    Divider,
} from '@mui/material'
import { NetworkDiagram } from './NetworkDiagram'
import { LayerEditor } from './LayerEditor'
import { validateArchitecture } from '../utils/architectureValidation'

interface NetworkArchitecture {
    [key: string]: {
        n_inputs: number
        n_neurons: number
        activation: string | null
    }
}

const defaultArchitecture: NetworkArchitecture = {
    input: {
        n_inputs: 2,
        n_neurons: 3,
        activation: null
    },
    output: {
        n_inputs: 3,
        n_neurons: 3,
        activation: 'softmax'
    }
}

const normalizeArchitectureLayers = (arch: NetworkArchitecture): NetworkArchitecture => {
    const layers = Object.entries(arch)
    if (layers.length < 2) return arch

    const normalized: NetworkArchitecture = {}
    
    // Set first layer as input
    const [_, firstLayer] = layers[0]
    normalized['input'] = firstLayer

    // Keep middle layers as-is
    layers.slice(1, -1).forEach(([_, config], idx) => {
        normalized[`hidden${idx + 1}`] = config
    })

    // Set last layer as output
    const [__, lastLayer] = layers[layers.length - 1]
    normalized['output'] = lastLayer

    return normalized
}

export const NetworkArchitectureEditor = () => {
    const [architecture, setArchitecture] = useState<NetworkArchitecture>(defaultArchitecture)
    const [jsonError, setJsonError] = useState<string>('')
    const [jsonText, setJsonText] = useState(JSON.stringify(architecture, null, 2))

    useEffect(() => {
        setJsonText(JSON.stringify(architecture, null, 2))
    }, [architecture])

    const handleJsonChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        const value = event.target.value
        setJsonText(value)
        
        try {
            const parsed = JSON.parse(value)
            const normalized = normalizeArchitectureLayers(parsed)
            setArchitecture(normalized)
            
            const validationErrors = validateArchitecture(normalized)
            if (validationErrors.length > 0) {
                setJsonError(validationErrors.map(err => err.message).join('\n'))
            } else {
                setJsonError('')
                const editor = document.querySelector('.json-editor')
                editor?.classList.add('valid-flash')
                setTimeout(() => editor?.classList.remove('valid-flash'), 1000)
            }
        } catch (err) {
            setJsonError(err instanceof Error ? err.message : 'Invalid JSON format')
        }
    }

    const handleExport = () => {
        const blob = new Blob([JSON.stringify(architecture, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'network-architecture.json'
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
    }

    const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (!file) return

        const reader = new FileReader()
        reader.onload = (e) => {
            try {
                const imported = JSON.parse(e.target?.result as string)
                const normalized = normalizeArchitectureLayers(imported)
                setArchitecture(normalized)
                
                const validationErrors = validateArchitecture(normalized)
                if (validationErrors.length > 0) {
                    setJsonError(validationErrors.map(err => err.message).join('\n'))
                } else {
                    setJsonError('')
                    const editor = document.querySelector('.json-editor')
                    editor?.classList.add('valid-flash')
                    setTimeout(() => editor?.classList.remove('valid-flash'), 1000)
                }
            } catch (err) {
                setJsonError('Failed to parse imported file')
            }
        }
        reader.readAsText(file)
    }

    const calculateEditorRows = () => {
        const layerCount = Object.keys(architecture).length
        // Each layer takes ~4 lines in JSON format, plus 2 for brackets and spacing
        return Math.max(10, layerCount * 4 + 2)
    }

    return (
        <Paper elevation={3} sx={{ p: 3, my: 2 }}>
            <Typography variant="h6" gutterBottom>
                Network Architecture
            </Typography>

            <Box sx={{
                display: 'flex',
                flexDirection: { xs: 'column', md: 'row' },
                gap: 3,
                mb: 3
            }}>
                <Box sx={{ flex: 1 }}>
                    <TextField
                        className="json-editor"
                        fullWidth
                        multiline
                        rows={calculateEditorRows()}
                        variant="outlined"
                        value={jsonText}
                        onChange={handleJsonChange}
                        error={Boolean(jsonError)}
                        helperText={jsonError}
                        sx={{
                            fontFamily: 'monospace',
                            '& .MuiInputBase-input': {
                                fontFamily: 'monospace',
                            },
                            height: '100%',
                            '& .MuiInputBase-root': {
                                height: '100%'
                            }
                        }}
                    />
                </Box>

                <Box sx={{ flex: 1 }}>
                    <LayerEditor
                        architecture={architecture}
                        onChange={setArchitecture}
                    />
                </Box>
            </Box>

            <Box sx={{ 
                display: 'flex', 
                gap: 2, 
                flexWrap: 'wrap',
                mb: 3
            }}>
                <Button
                    variant="outlined"
                    component="label"
                >
                    Import JSON
                    <input
                        type="file"
                        hidden
                        accept=".json"
                        onChange={handleImport}
                    />
                </Button>
                <Button
                    variant="outlined"
                    onClick={handleExport}
                >
                    Export JSON
                </Button>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={() => setArchitecture(defaultArchitecture)}
                >
                    Reset to Default
                </Button>
                <Button
                    variant="contained"
                    color="secondary"
                    onClick={() => {
                        const errors = validateArchitecture(architecture)
                        if (errors.length > 0) {
                            setJsonError(errors.map(err => err.message).join('\n'))
                        } else {
                            setJsonError('')
                        }
                    }}
                >
                    Validate Architecture
                </Button>
            </Box>

            <Divider sx={{ my: 3 }} />
            <NetworkDiagram architecture={architecture} />
        </Paper>
    )
} 