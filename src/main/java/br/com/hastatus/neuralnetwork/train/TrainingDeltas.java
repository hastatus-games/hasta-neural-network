package br.com.hastatus.neuralnetwork.train;


/**
 * This class is used to store intermediary values during training that will be used to adjust weights and bias
 */
public class TrainingDeltas {

    double[] inputs;

    double[][] deltas;



    public TrainingDeltas(double[] inputs, int totalLayers) {
        this.inputs = inputs;
        deltas = new double[totalLayers][];

    }

    public void initDeltaLayer(int layerIndex, int outputSize){
        deltas[layerIndex] = new double[outputSize];

    }

    public void addDelta(int layerIndex, int neuronIndex, double value) {
        deltas[layerIndex][neuronIndex] = value;
    }


    public double getDelta(int layerIndex, int neuronIndex) {
        return deltas[layerIndex][neuronIndex];
    }

    public double[] getInputs() {
        return inputs;
    }
}
