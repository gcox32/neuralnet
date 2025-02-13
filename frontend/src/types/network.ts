export const ACTIVATION_COLORS = {
  'relu': '#ff6b6b',
  'sigmoid': '#4ecdc4',
  'tanh': '#45b7d1',
  'softmax': '#96ceb4',
  'linear': '#646cff',
  'null': '#646cff'
} as const;

export type ActivationType = keyof typeof ACTIVATION_COLORS;

export const ACTIVATION_OPTIONS = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', null] as const;

export interface NetworkArchitecture {
  [key: string]: {
    n_inputs: number
    n_neurons: number
    activation: ActivationType | null
  }
} 