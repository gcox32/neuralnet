import { ACTIVATION_OPTIONS } from '../types/network'

interface ValidationError {
  message: string
  path?: string
}

export const validateArchitecture = (architecture: any): ValidationError[] => {
  const errors: ValidationError[] = []

  if (!architecture || typeof architecture !== 'object') {
    return [{ message: 'Architecture must be an object' }]
  }

  // Required layers
  if (!architecture.input) errors.push({ message: 'Missing input layer' })
  if (!architecture.output) errors.push({ message: 'Missing output layer' })

  // Validate each layer
  Object.entries(architecture).forEach(([layerName, layer]: [string, any]) => {
    const basePath = `layers.${layerName}`

    // Check required keys
    const requiredKeys = ['n_inputs', 'n_neurons', 'activation']
    requiredKeys.forEach(key => {
      if (!(key in layer)) {
        errors.push({
          message: `Missing required key "${key}" in layer "${layerName}"`,
          path: `${basePath}.${key}`
        })
      }
    })

    // Validate n_inputs
    if ('n_inputs' in layer) {
      if (typeof layer.n_inputs !== 'number' || layer.n_inputs <= 0) {
        errors.push({ 
          message: 'n_inputs must be a positive number',
          path: `${basePath}.n_inputs`
        })
      }
    }

    // Validate n_neurons
    if ('n_neurons' in layer) {
      if (typeof layer.n_neurons !== 'number' || layer.n_neurons <= 0) {
        errors.push({ 
          message: 'n_neurons must be a positive number',
          path: `${basePath}.n_neurons`
        })
      }
    }

    // Validate activation
    if ('activation' in layer) {
      const validActivations = ACTIVATION_OPTIONS.filter(option => option !== null)
      const activation = layer.activation?.toLowerCase() || null
      
      if (activation !== null && !validActivations.includes(activation)) {
        errors.push({
          message: `Invalid activation function "${layer.activation}". Must be one of: ${validActivations.join(', ')} or null`,
          path: `${basePath}.activation`
        })
      }
    }

    // Validate layer connections
    if (layerName !== 'input') {
      const prevLayerName = layerName.startsWith('hidden') 
        ? `hidden${parseInt(layerName.replace('hidden', '')) - 1}` 
        : 'input'
      
      const prevLayer = architecture[prevLayerName]
      if (prevLayer && layer.n_inputs !== prevLayer.n_neurons) {
        errors.push({
          message: `Layer ${layerName} n_inputs (${layer.n_inputs}) must match previous layer n_neurons (${prevLayer.n_neurons})`,
          path: `${basePath}.n_inputs`
        })
      }
    }
  })

  return errors
} 