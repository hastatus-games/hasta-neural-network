package br.com.hastatus.neuralnetwork.base;

import java.util.ArrayList;
import java.util.List;

/**
 * The fundamental building block of a neural network. A neuron takes one or more inputs, each of which is multiplied by a respective weight.
 * The results of these multiplications are then summed together and a bias value is added to this sum.
 * This final value is passed through an activation function to produce the neuron's output.
 */
public abstract class Neuron {

    private static final double INITIAL_WEIGHT_VALUE = 0.5;
    private static final double INITIAL_BIAS_VALUE = 0.1;

    private final List<Double> weights;
    private double bias;


    /**
     * Initialize the neuron with a total number of inputs to connect.
     * Each input is associated with a weight. The default value for each weight is defined in {@link #INITIAL_WEIGHT_VALUE} and for bias is {@link #INITIAL_BIAS_VALUE}
     *
     * @param inputs The total number of inputs to connect each one with a specific weight
     */
    public Neuron(int inputs) {
        this.weights = new ArrayList<>(inputs);

        for (int i = 0; i < inputs; i++) {
            this.weights.add(i, INITIAL_WEIGHT_VALUE);
        }
        this.bias = INITIAL_BIAS_VALUE;
    }




    public double activate(double[] inputs) {

        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights.get(i);
        }
        sum += this.bias;
        return activateFunction(sum);
    }


    public void adjustWeight(int index, double delta) {
        Double value = this.weights.get(index);
        this.weights.set(index, value-delta);
    }

    public void adjustBias(double delta) {
        this.bias -= delta;
    }

    public double getWeight(int index) {
        return this.weights.get(index);
    }

    abstract double activateFunction(double value);

    abstract double deriveActivationFunction(double value);
}

