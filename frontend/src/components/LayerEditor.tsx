import {
  Box,
  TextField,
  Select,
  MenuItem,
  IconButton,
  FormControl,
  InputLabel,
  Typography,
  Button,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import DeleteIcon from '@mui/icons-material/Delete'

interface LayerConfig {
  n_inputs: number
  n_neurons: number
  activation: string | null
}

interface LayerEditorProps {
  architecture: { [key: string]: LayerConfig }
  onChange: (newArchitecture: { [key: string]: LayerConfig }) => void
}

const ACTIVATION_OPTIONS = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', null]

export const LayerEditor = ({ architecture, onChange }: LayerEditorProps) => {
  const handleLayerChange = (layerName: string, field: keyof LayerConfig, value: any) => {
    const newArchitecture = {
      ...architecture,
      [layerName]: {
        ...architecture[layerName],
        [field]: field === 'activation' 
          ? (value ? value.toLowerCase() : null)
          : Number(value)
      }
    }
    onChange(newArchitecture)
  }

  const removeLayer = (layerName: string) => {
    if (layerName === 'input' || layerName === 'output') return
    const newArchitecture = { ...architecture }
    delete newArchitecture[layerName]
    onChange(newArchitecture)
  }

  const addLayer = () => {
    const layers = Object.entries(architecture)
    const outputLayer = layers.pop()!
    const prevLayer = layers[layers.length - 1][1]
    
    const newArchitecture = { ...architecture }
    delete newArchitecture['output']
    
    const newLayerName = `hidden${layers.length}`
    newArchitecture[newLayerName] = {
      n_inputs: prevLayer.n_neurons,
      n_neurons: prevLayer.n_neurons,
      activation: 'relu'
    }
    newArchitecture['output'] = outputLayer[1]
    
    onChange(newArchitecture)
  }

  const moveLayer = (layerName: string, direction: 'up' | 'down') => {
    const layers = Object.entries(architecture)
    const currentIndex = layers.findIndex(([name]) => name === layerName)
    if (
      (direction === 'up' && currentIndex <= 0) || 
      (direction === 'down' && currentIndex >= layers.length - 1)
    ) return

    const newIndex = direction === 'up' ? currentIndex - 1 : currentIndex + 1
    const newArchitecture: { [key: string]: LayerConfig } = {}
    
    layers.forEach(([name, config], index) => {
      if (index === currentIndex) return
      if (index === newIndex) {
        if (direction === 'up') {
          newArchitecture[layerName] = architecture[layerName]
          newArchitecture[name] = config
        } else {
          newArchitecture[name] = config
          newArchitecture[layerName] = architecture[layerName]
        }
      } else {
        newArchitecture[name] = config
      }
    })

    onChange(newArchitecture)
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        Layer Configuration
      </Typography>
      
      {Object.entries(architecture).map(([layerName, config], index, array) => (
        <Box 
          key={layerName} 
          sx={{ 
            display: 'flex', 
            gap: 2, 
            mb: 2,
            alignItems: 'center'
          }}
        >
          <Box sx={{ width: '15%' }}>
            <Typography variant="body1">
              {layerName}
            </Typography>
          </Box>
          
          <Box sx={{ width: '25%' }}>
            <TextField
              label="Neurons"
              type="number"
              value={config.n_neurons}
              onChange={(e) => handleLayerChange(layerName, 'n_neurons', e.target.value)}
              fullWidth
            />
          </Box>
          
          <Box sx={{ width: '35%' }}>
            <FormControl fullWidth>
              <InputLabel>Activation</InputLabel>
              <Select
                value={config.activation?.toLowerCase() || ''}
                label="Activation"
                onChange={(e) => handleLayerChange(layerName, 'activation', e.target.value)}
              >
                {ACTIVATION_OPTIONS.map((activation) => (
                  <MenuItem key={activation || 'null'} value={activation || ''}>
                    {activation || 'None'}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton 
              onClick={() => moveLayer(layerName, 'up')}
              disabled={index === 0}
            >
              <ArrowUpwardIcon />
            </IconButton>
            <IconButton 
              onClick={() => moveLayer(layerName, 'down')}
              disabled={index === array.length - 1}
            >
              <ArrowDownwardIcon />
            </IconButton>
            <IconButton
              onClick={() => removeLayer(layerName)}
              disabled={layerName === 'input' || layerName === 'output'}
              color="error"
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        </Box>
      ))}
      
      <Button
        startIcon={<AddIcon />}
        variant="outlined"
        onClick={addLayer}
        sx={{ mt: 2 }}
      >
        Add Hidden Layer
      </Button>
    </Box>
  )
} 