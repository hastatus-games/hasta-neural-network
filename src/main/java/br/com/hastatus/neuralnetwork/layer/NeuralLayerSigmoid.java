package br.com.hastatus.neuralnetwork.layer;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralLayerSigmoid implements NeuralLayer {

    private final List<Neuron> neurons;
    private double[] lastInputs; // Store the last inputs received


    public NeuralLayerSigmoid(int numNeurons, int numInputsPerNeuron) {
        this.neurons = new ArrayList<>();

        for (int i = 0; i < numNeurons; i++) {
            this.neurons.add(new NeuronSigmoid(numInputsPerNeuron));
        }
    }



    public double[] forward(double[] inputs) {
        this.lastInputs = Arrays.copyOf(inputs, inputs.length);
        double[] outputs = new double[this.neurons.size()];

        for (int i = 0; i < this.neurons.size(); i++) {
            outputs[i] = this.neurons.get(i).activate(inputs);
        }
        return outputs;
    }


    public void adjustWeights(double[] inputs, double[] weightDeltas, double learningRate) {
        if(learningRate == 0) {
            throw new RuntimeException("LearningRate must be != 0");
        }
        int neuronSize = neurons.size();
        int inputLength = inputs.length;

        if(inputLength != neuronSize || weightDeltas.length!=neuronSize) {
            throw new RuntimeException(String.format("Invalid number of inputs and/or weightDeltas, expected: %d", neuronSize));
        }

        for (int i = 0; i < neuronSize; i++) {
            Neuron neuron = neurons.get(i);
            for (int j = 0; j < inputLength; j++) {
                neuron.adjustWeight(j, weightDeltas[i] * inputs[j] * learningRate);
            }
        }
    }

    public void adjustBias(double[] biasDeltas) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).adjustBias(biasDeltas[i]);
        }
    }

    public Neuron getNeuron(int index) {
        return neurons.get(index);
    }

    public int getTotalNeurons() {
        return neurons.size();
    }


    public double[] getLastInputs() {
        return lastInputs;
    }
}
