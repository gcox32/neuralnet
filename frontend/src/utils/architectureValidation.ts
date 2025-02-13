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

    if (!layer.n_inputs || typeof layer.n_inputs !== 'number') {
      errors.push({ 
        message: 'n_inputs must be a positive number',
        path: `${basePath}.n_inputs`
      })
    }

    if (!layer.n_neurons || typeof layer.n_neurons !== 'number') {
      errors.push({ 
        message: 'n_neurons must be a positive number',
        path: `${basePath}.n_neurons`
      })
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