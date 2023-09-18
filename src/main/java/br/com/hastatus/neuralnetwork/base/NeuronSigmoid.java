package br.com.hastatus.neuralnetwork.base;

/**
 * Implementation of a neuron using a Sigmoid activation function.
 * The Sigmoid function is a mathematical function that produces an "S" shaped curve;
 * it is commonly used in neural networks to introduce nonlinearity in the model and to map any input into a value between 0 and 1.
 */
public class NeuronSigmoid extends Neuron {

    public NeuronSigmoid(int inputs) {
        super(inputs);
    }

    @Override
    double activateFunction(double value) {
        return 1.0 / (1.0 + Math.exp(-value)); //Sigmoid
    }


    @Override
    public double deriveActivationFunction(double value){
        return deriveSigmoid(value); // Sigmoid derivative
    }


    public static double deriveSigmoid(double value){
        return value * (1 - value);
    }
}
