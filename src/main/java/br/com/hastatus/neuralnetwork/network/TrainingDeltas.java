package br.com.hastatus.neuralnetwork.network;

public class TrainingDeltas {

    double[] entrada;

    double[][] deltas;


    public TrainingDeltas(double[] entrada, int totalLayers) {
        this.entrada = entrada;
        deltas = new double[totalLayers][];

    }

    public void initDeltaLayer(int layerIndex, int outputSize){
        // Calculate deltas for the output layer
        int lastLayerIndex = deltas.length - 1;
        deltas[layerIndex] = new double[outputSize];

    }

    public void addDelta(int layerIndex, int neuronIndex, double value) {
        deltas[layerIndex][neuronIndex] = value;
    }

    public double getDelta(int layerIndex, int neuronIndex) {
        return deltas[layerIndex][neuronIndex];
    }

    public double[] getEntrada() {
        return entrada;
    }
}
