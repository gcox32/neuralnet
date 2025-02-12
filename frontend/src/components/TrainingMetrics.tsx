import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface MetricsProps {
  data: {
    iterations: number[];
    loss: number[];
    accuracy: number[];
    learning_rate: number[];
  };
}

export const TrainingMetrics = ({ data }: MetricsProps) => {
  const chartData = data.iterations.map((iter, idx) => ({
    iteration: iter,
    loss: data.loss[idx],
    accuracy: data.accuracy[idx],
    learning_rate: data.learning_rate[idx]
  }));

  return (
    <div className="metrics-container">
      <LineChart width={600} height={300} data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="iteration" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#8884d8" />
        <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" />
      </LineChart>
    </div>
  );
};
